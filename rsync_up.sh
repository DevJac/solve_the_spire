#!/usr/bin/env bash

echo 'Syncing solve_the_spire'
rsync -ac --info=progress2 ~/Code/julia/solve_the_spire root@$DROPLET_IP:

echo 'Syncing GOG Games'
rsync -ac --info=progress2 ~/GOG\ Games root@$DROPLET_IP:

echo 'Syncing ModTheSpire'
rsync -ac --info=progress2 ~/.config/ModTheSpire root@$DROPLET_IP:.config
