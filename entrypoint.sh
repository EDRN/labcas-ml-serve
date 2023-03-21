#!/bin/sh -x
#
# Docker entrypoint for LabCAS ML Serve
#
# This just executes `ray_start.py` and if it exits successfully, goes into a spin loop.
# A better approach would be for `ray_start.py` to stay in the foreground, but I don't
# know enough Ray Serve to do that yet.

cd /usr/src/app
echo "🤯 Here is what I find"
ls -l src
echo "🧐 See, it is there"
/usr/local/bin/python src/ray_start.py </dev/null
rc=$?

if [ $rc -ne 0 ]; then
    echo "🤒 ray_start failed with $rc; container exiting" 1>&2
    exit -1
else
    # A container shouldn't exit, but ray_start.py exits whether successful or not.
    # So, start spinning!
    while :; do
        sleep 999999
    done
fi
