Gaze Game Alpha Test
====================

Keep these files together:

- Gaze Game.app
- relay_urls.txt, if it is included

Do not move relay_urls.txt inside the app. It should stay in this same folder as
Gaze Game.app.

If You Are Testing Remotely
---------------------------

1. Install Tailscale for Mac:
   https://tailscale.com/download/mac

2. Sign in or accept the invite from the person running the test.

3. Wait until Tailscale says it is connected.

4. Leave Tailscale running while using Gaze Game.

Open Gaze Game
--------------

1. Right-click Gaze Game.app.
2. Click Open.
3. If macOS asks whether you are sure, click Open again.
4. When macOS asks for Camera access, click Allow.

Join A Room
-----------

If you are the first person:

1. Type your name.
2. Click Create Room.
3. Send the room code to the other person.

If you are joining someone:

1. Type your name.
2. Type the room code they sent you.
3. Click Join Room.

What Should Happen
------------------

You should see a dark grid with your gaze cursor and the other person's cursor.

If It Does Not Work
-------------------

Relay unavailable:
The relay is unreachable. Make sure Tailscale is connected and relay_urls.txt is still
next to Gaze Game.app.

Camera unavailable:
Open System Settings > Privacy & Security > Camera and enable Gaze Game. Then reopen
the app.

App will not open:
Right-click Gaze Game.app and choose Open. Do not double-click the first time.

Gaze is inaccurate:
This is expected for early testing if the model was not trained on your MacBook and your
eyes. The network test is still working if your cursor appears and moves.
