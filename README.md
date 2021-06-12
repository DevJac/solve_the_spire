A Slay the Spire AI
===================

Setup
-----

Install Slay the Spire. Install [Communication Mod](https://github.com/ForgottenArbiter/CommunicationMod), and optinally [SuperFastMode](https://github.com/Skrelpoid/SuperFastMode). I installed them from the Steam Workshop.

Setup this Julia environment as usual.

Usage
-----

Configure Communication Mod to run `relay.jl`, and then run `agent.jl`.

Cloud Setup
-----------

`dnf install htop git screen kernel-modules* java-1.8.0 wget xrandr xorg-x11-server-Xvfb xpra vim`

Useful `screen` commands:
`screen -S name`
`screen -r name`
`screen -list`

In `screen`, do ctrl-a d to detach.

In `~/.bash_profile`:
```
export PUSHOVER_USER_KEY=abcdef
export PUSHOVER_STS_API_TOKEN=abcdef
export JULIA_LOAD_PATH=src:
modprobe snd-dummy
```
