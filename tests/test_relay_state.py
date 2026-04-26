import unittest

from shared_gaze.protocol import clamp01, decode, encode
from shared_gaze.relay_server import (
    MAX_CLIENTS_PER_ROOM,
    MAX_WAVE_DURATION_MS,
    RelayClient,
    RelayState,
    clean_duration_ms,
    clean_name,
    clean_room,
    clean_wave_targets,
)


class RelayProtocolTests(unittest.TestCase):
    def test_json_messages_round_trip_as_objects(self) -> None:
        payload = {"type": "cursor", "x": 0.25, "tracking": True}

        self.assertEqual(decode(encode(payload)), payload)

    def test_clamp01_tolerates_bad_values(self) -> None:
        self.assertEqual(clamp01(-1), 0.0)
        self.assertEqual(clamp01(2), 1.0)
        self.assertEqual(clamp01("bad"), 0.0)


class RelayStateTests(unittest.TestCase):
    def test_room_join_leave_tracks_clients_by_room(self) -> None:
        state = RelayState()
        client = RelayClient(id="client-a", websocket=None)

        state.join_room(client, "ROOM-A")
        self.assertEqual(client.room, "ROOM-A")
        self.assertIn("client-a", state.rooms["ROOM-A"])

        state.join_room(client, "ROOM-B")
        self.assertNotIn("ROOM-A", state.rooms)
        self.assertIn("client-a", state.rooms["ROOM-B"])

        state.leave_room(client)
        self.assertIsNone(client.room)
        self.assertNotIn("ROOM-B", state.rooms)

    def test_room_capacity_rejects_extra_clients(self) -> None:
        state = RelayState()
        clients = [
            RelayClient(id=f"client-{index}", websocket=None)
            for index in range(MAX_CLIENTS_PER_ROOM)
        ]
        for client in clients:
            self.assertTrue(state.can_join_room(client, "ROOM"))
            state.join_room(client, "ROOM")

        extra_client = RelayClient(id="client-extra", websocket=None)
        self.assertFalse(state.can_join_room(extra_client, "ROOM"))
        self.assertTrue(state.can_join_room(clients[0], "ROOM"))

    def test_active_wave_is_room_scoped_and_reused_until_expiry(self) -> None:
        state = RelayState()
        client_a = RelayClient(id="client-a", websocket=None, name="Ada")
        client_b = RelayClient(id="client-b", websocket=None, name="Ben")
        state.join_room(client_a, "ROOM-A")
        state.join_room(client_b, "ROOM-B")

        wave_a = state.start_wave(
            "ROOM-A",
            client_a,
            {
                "seed": "seed-a",
                "duration_ms": 999999,
                "targets": [{"id": "t1", "x": -1, "y": 2}],
            },
        )
        reused = state.start_wave("ROOM-A", client_a, {"seed": "seed-b"})
        wave_b = state.start_wave("ROOM-B", client_b, {"seed": "seed-b"})

        self.assertEqual(reused["id"], wave_a["id"])
        self.assertNotEqual(wave_a["id"], wave_b["id"])
        self.assertEqual(wave_a["duration_ms"], MAX_WAVE_DURATION_MS)
        self.assertEqual(wave_a["targets"], [{"id": "t1", "x": 0.0, "y": 1.0}])

    def test_wave_score_updates_are_sanitized(self) -> None:
        state = RelayState()
        client = RelayClient(id="client-a", websocket=None, name="Ada", color=(1, 2, 3))
        state.join_room(client, "ROOM")
        wave = state.start_wave("ROOM", client, {"seed": "seed"})

        score = state.update_wave_score("ROOM", client, wave["id"], 10000, "target-a")

        self.assertIsNotNone(score)
        self.assertEqual(score["score"], 9999)
        self.assertEqual(score["name"], "Ada")
        self.assertEqual(score["color"], [1, 2, 3])
        self.assertEqual(score["target_id"], "target-a")
        self.assertIsNone(state.update_wave_score("ROOM", client, "wrong-wave", 10, "target-a"))

    def test_cleaners_bound_user_supplied_payloads(self) -> None:
        self.assertEqual(
            clean_room("  ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3),
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ  ABCDEFGHIJKLMNOPQRST",
        )
        self.assertEqual(clean_name("x" * 40), "x" * 32)
        self.assertEqual(clean_duration_ms(1), 5000)
        self.assertEqual(clean_duration_ms(999999), MAX_WAVE_DURATION_MS)
        self.assertEqual(
            clean_wave_targets([{"x": -4, "y": 5}, "bad"]),
            [{"id": "enemy-0", "x": 0.0, "y": 1.0}],
        )


if __name__ == "__main__":
    unittest.main()
