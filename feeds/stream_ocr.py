"""
Stream OCR feed — extracts live game state from Twitch/YouTube broadcasts.

Captures frames from esports streams and uses macOS Vision OCR to read
scores, kills, gold, timers, and team names from the broadcast overlay.

Works completely in the background — no browser needed.
Pipeline: streamlink → ffmpeg → OCR → game events

Supports: LoL, Valorant (any game with a consistent broadcast overlay)
"""
import asyncio
import logging
import os
import re
import subprocess
import time
from typing import Optional

import cv2

from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

STREAMLINK = os.path.expanduser("~/Library/Python/3.9/bin/streamlink")
FFMPEG = "/opt/homebrew/bin/ffmpeg"


class StreamOCRFeed(GameFeed):
    """Extracts live game state from Twitch streams via OCR."""

    CAPTURE_INTERVAL = 8  # seconds between frame captures

    def __init__(self, game: str):
        super().__init__(game)
        self._streams: dict[str, dict] = {}  # match_id -> {url, team_a, team_b, ...}
        self._poll_task = None
        self._last_ocr: dict[str, dict] = {}  # match_id -> last OCR results

    async def connect(self):
        self._connected = True
        self._connect_time = time.time()
        self._running = True
        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game=self.game, team="neutral",
            description=f"Stream OCR feed ready for {self.game}", impact=0.0,
        ))
        logger.info(f"Stream OCR feed ready for {self.game}")
        self._poll_task = asyncio.create_task(self._safe_run(self._ocr_loop(), "stream_ocr"))

    async def _safe_run(self, coro, name):
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Task {name} CRASHED: {e}", exc_info=True)

    async def disconnect(self):
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
        self._connected = False

    async def subscribe_match(self, match_id: str):
        pass

    async def unsubscribe_match(self, match_id: str):
        self._streams.pop(match_id, None)

    async def get_live_matches(self) -> list[MatchState]:
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        return list(self._streams.keys())

    def add_stream(self, match_id: str, stream_url: str, team_a: str, team_b: str):
        """Add a stream URL to monitor for a match."""
        self._streams[match_id] = {
            "url": stream_url,
            "team_a": team_a,
            "team_b": team_b,
        }
        if match_id not in self._matches:
            self._matches[match_id] = MatchState(
                match_id=match_id, game=self.game,
                team_a=team_a, team_b=team_b,
                is_live=True, started_at=time.time(),
                total_maps=3, extra={"ocr_source": stream_url},
            )
        logger.info(f"[OCR] Monitoring: {team_a} vs {team_b} via {stream_url}")

    def estimate_win_probability(self, state: MatchState) -> float:
        maps_needed = (state.total_maps // 2) + 1
        if state.score_a >= maps_needed:
            return 0.99
        if state.score_b >= maps_needed:
            return 0.01
        a_needs = maps_needed - state.score_a
        b_needs = maps_needed - state.score_b
        raw = max(0.01, min(0.99, b_needs / (a_needs + b_needs)))
        prior = state.extra.get("prior_prob_a")
        if prior is not None:
            from prior import apply_prior
            return apply_prior(raw, prior)
        return raw

    # ─── OCR Loop ────────────────────────────────────────────────────────────

    async def _ocr_loop(self):
        """Capture frames and OCR them periodically."""
        while self._running:
            for match_id, stream_info in list(self._streams.items()):
                try:
                    loop = asyncio.get_event_loop()
                    ocr_data = await loop.run_in_executor(
                        None, self._capture_and_ocr, stream_info["url"]
                    )
                    if ocr_data:
                        self._process_ocr(match_id, ocr_data)
                except Exception as e:
                    logger.warning(f"[OCR] Error for {match_id}: {e}")

            await asyncio.sleep(self.CAPTURE_INTERVAL)

    def _capture_and_ocr(self, stream_url: str) -> Optional[list]:
        """Capture a frame from stream and run OCR. Returns list of text blocks."""
        try:
            # Get stream URL via streamlink
            result = subprocess.run(
                [STREAMLINK, "--stream-url", stream_url, "480p,best"],
                capture_output=True, text=True, timeout=20,
            )
            raw_url = result.stdout.strip()
            if not raw_url:
                return None

            # Capture single frame with ffmpeg
            frame_path = f"/tmp/ocr_frame_{hash(stream_url) % 10000}.jpg"
            subprocess.run([
                FFMPEG, "-i", raw_url,
                "-frames:v", "1", "-y", frame_path,
            ], capture_output=True, timeout=15)

            if not os.path.exists(frame_path):
                return None

            # OCR using macOS Vision framework
            return self._ocr_frame(frame_path)

        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            logger.debug(f"[OCR] Capture failed: {e}")
            return None

    def _ocr_frame(self, frame_path: str) -> list:
        """Run macOS Vision OCR on a frame. Returns list of {text, x, y, confidence}."""
        try:
            import Vision
            from Foundation import NSData
            from Quartz import CGImageSourceCreateWithData, CGImageSourceCreateImageAtIndex

            with open(frame_path, "rb") as f:
                img_data = f.read()

            ns_data = NSData.dataWithBytes_length_(img_data, len(img_data))
            source = CGImageSourceCreateWithData(ns_data, None)
            if not source:
                return []
            cg_image = CGImageSourceCreateImageAtIndex(source, 0, None)
            if not cg_image:
                return []

            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)

            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
            success = handler.performRequests_error_([request], None)

            results = []
            for r in (request.results() or []):
                box = r.boundingBox()
                results.append({
                    "text": r.text(),
                    "x": box.origin.x,
                    "y": box.origin.y,
                    "w": box.size.width,
                    "h": box.size.height,
                    "confidence": r.confidence(),
                })
            return results

        except Exception as e:
            logger.debug(f"[OCR] Vision error: {e}")
            return []

    def _process_ocr(self, match_id: str, ocr_blocks: list):
        """Parse OCR results into game state updates."""
        state = self._matches.get(match_id)
        if not state:
            return

        # Detect if we're in-game or lobby
        # Lobby indicators: "MATCH", "TODAY", "WEEK", "DAY", schedule times like "18:00"
        # In-game indicators: timer at top center, gold values with "K", kill numbers
        all_text = " ".join(b["text"] for b in ocr_blocks).lower()
        lobby_keywords = ["today's match", "match up", "week ", "day ", "upcoming"]
        is_lobby = any(kw in all_text for kw in lobby_keywords)
        if is_lobby:
            return  # skip lobby screens

        # Extract scores from OCR text
        # Look for patterns like "8" and "7" near the center-top (scoreboard area)
        # Also look for round info, map names, timers

        scores = []
        round_info = ""
        map_name = ""
        timer = ""

        for block in ocr_blocks:
            text = block["text"].strip()
            y = block["y"]  # 0=bottom, 1=top in Vision coordinates
            x = block["x"]

            # Score-like numbers in the top area (y > 0.5)
            if y > 0.5 and text.isdigit() and int(text) < 100:
                scores.append({"value": int(text), "x": x, "y": y})

            # Round info
            if "round" in text.lower():
                round_info = text

            # Map names (common Valorant/LoL map names)
            for map_n in ["bind", "haven", "split", "ascent", "icebox", "breeze", "fracture",
                          "pearl", "lotus", "sunset", "abyss",
                          "summoner", "rift"]:
                if map_n in text.lower():
                    map_name = text

            # Timer (MM:SS format)
            if re.match(r"^\d{1,2}:\d{2}$", text):
                timer = text

            # Score with dash (e.g., "9-13")
            score_match = re.match(r"(\d{1,2})\s*[-–]\s*(\d{1,2})", text)
            if score_match and y > 0.7:
                s1, s2 = int(score_match.group(1)), int(score_match.group(2))
                state.extra["ocr_map_score"] = f"{s1}-{s2}"

        # Try to determine round score from two score numbers near center
        if len(scores) >= 2:
            # Sort by x position — left score is team A, right is team B
            scores.sort(key=lambda s: s["x"])
            center_scores = [s for s in scores if 0.3 < s["x"] < 0.7]
            if len(center_scores) >= 2:
                new_a = center_scores[0]["value"]
                new_b = center_scores[1]["value"]
                old_a = state.round_score_a
                old_b = state.round_score_b

                if (new_a, new_b) != (old_a, old_b) and (new_a + new_b) > 0:
                    state.round_score_a = new_a
                    state.round_score_b = new_b
                    state.extra["round_score"] = f"{new_a}-{new_b}"
                    state.extra["ocr_timer"] = timer
                    state.extra["ocr_map"] = map_name

                    if old_a + old_b > 0:  # not first read
                        winner = "a" if new_a > old_a else "b"
                        winner_name = state.team_a if winner == "a" else state.team_b

                        old_prob = state.win_probability_a
                        state.win_probability_a = self.estimate_win_probability(state)

                        logger.info(f"[OCR] {state.team_a} {new_a}-{new_b} {state.team_b} | {map_name} {timer}")

                        self._emit(GameEvent(
                            event_type=EventType.ROUND_END,
                            match_id=match_id, game=self.game, team=winner,
                            description=f"Round to {winner_name} ({new_a}-{new_b}) [OCR from stream]",
                            impact=state.win_probability_a - old_prob,
                            match_state=state,
                        ))
                    else:
                        logger.info(f"[OCR] First read: {state.team_a} {new_a}-{new_b} {state.team_b}")

        # LoL-specific: extract kills and gold from top bar
        if self.game == "lol":
            self._process_lol_ocr(match_id, ocr_blocks, state)

        prev = self._last_ocr.get(match_id, {})
        self._last_ocr[match_id] = {
            "round_a": state.round_score_a,
            "round_b": state.round_score_b,
            "time": time.time(),
        }

    def _process_lol_ocr(self, match_id: str, ocr_blocks: list, state: MatchState):
        """Parse LoL broadcast overlay — kills, gold, timer from top bar."""
        kills_found = []
        gold_found = []
        timer = ""
        series_score = ""

        for block in ocr_blocks:
            text = block["text"].strip()
            y = block["y"]
            x = block["x"]

            # Top bar only (y > 0.85 in Vision coords)
            if y < 0.85:
                continue

            # Timer (MM:SS)
            if re.match(r"^\d{1,2}:\d{2}$", text) and x > 0.4 and x < 0.6:
                timer = text

            # Series score (e.g., "3-0", "1-2")
            score_match = re.match(r"^(\d)-(\d)$", text)
            if score_match:
                series_score = text

            # Gold values (e.g., "82.8K", "2.8K")
            gold_match = re.match(r"([\d.]+)K", text.replace(" ", ""))
            if gold_match:
                gold_found.append({"value": float(gold_match.group(1)) * 1000, "x": x})

            # Kill counts — single or double digit numbers
            if text.isdigit() and int(text) < 100 and 0.3 < x < 0.7:
                kills_found.append({"value": int(text), "x": x})

        # Update state with extracted LoL data
        if timer:
            state.extra["ocr_timer"] = timer

        if series_score:
            parts = series_score.split("-")
            try:
                new_a, new_b = int(parts[0]), int(parts[1])
                old_a, old_b = state.score_a, state.score_b
                if (new_a, new_b) != (old_a, old_b) and new_a + new_b > old_a + old_b:
                    state.score_a = new_a
                    state.score_b = new_b

                    winner = "a" if new_a > old_a else "b"
                    winner_name = state.team_a if winner == "a" else state.team_b
                    old_prob = state.win_probability_a
                    state.win_probability_a = self.estimate_win_probability(state)

                    logger.info(f"[OCR LoL] {state.team_a} {new_a}-{new_b} {state.team_b} | series score change")

                    self._emit(GameEvent(
                        event_type=EventType.MAP_WIN,
                        match_id=match_id, game="lol", team=winner,
                        description=f"Game to {winner_name} (Series: {new_a}-{new_b}) [OCR]",
                        impact=state.win_probability_a - old_prob,
                        match_state=state,
                    ))
            except (ValueError, IndexError):
                pass

        # Gold lead
        if len(gold_found) >= 2:
            gold_found.sort(key=lambda g: g["x"])
            gold_a = gold_found[0]["value"]
            gold_b = gold_found[-1]["value"]
            state.extra["gold_a"] = int(gold_a)
            state.extra["gold_b"] = int(gold_b)
            state.extra["gold_lead"] = int(gold_a - gold_b)

        # Kills
        if len(kills_found) >= 2:
            kills_found.sort(key=lambda k: k["x"])
            center_kills = [k for k in kills_found if 0.3 < k["x"] < 0.7]
            if len(center_kills) >= 2:
                old_ka = state.extra.get("kills_a", 0)
                old_kb = state.extra.get("kills_b", 0)
                new_ka = center_kills[0]["value"]
                new_kb = center_kills[-1]["value"]

                if new_ka != old_ka or new_kb != old_kb:
                    state.extra["kills_a"] = new_ka
                    state.extra["kills_b"] = new_kb

                    if old_ka + old_kb > 0:
                        team = "a" if new_ka > old_ka else "b"
                        old_prob = state.win_probability_a
                        state.win_probability_a = self.estimate_win_probability(state)

                        gold_str = f" | Gold: {state.extra.get('gold_a',0):,} vs {state.extra.get('gold_b',0):,}" if state.extra.get("gold_a") else ""

                        logger.info(f"[OCR LoL] {state.team_a} {new_ka}-{new_kb} {state.team_b}{gold_str}")

                        self._emit(GameEvent(
                            event_type=EventType.SCORE_UPDATE,
                            match_id=match_id, game="lol", team=team,
                            description=f"Kills: {state.team_a} {new_ka}-{new_kb} {state.team_b}{gold_str} [OCR]",
                            impact=state.win_probability_a - old_prob,
                            match_state=state,
                        ))
