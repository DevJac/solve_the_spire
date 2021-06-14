#!/usr/bin/env bash

rsync -ac --info=progress2 root@$DROPLET_IP:solve_the_spire/tb_logs .
rsync -ac --info=progress2 root@$DROPLET_IP:solve_the_spire/models .
rsync -ac --info=progress2 root@$DROPLET_IP:solve_the_spire/log.txt .
