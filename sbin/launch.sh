#!/bin/sh
#
# Non-containerized entrypoint for LabCAS ML Serve
#
# This just executes `ray_start.py` and if it exits successfully, goes into a spin loop.
# A better approach would be for us to get Ray Serve started without the extra baggage and
# have it stay in the foreground. Yeah, this is ugly.
#
# We assume `python3` on the PATH is the correct Python 3.9 virtual environemnt with all
# dependencies to support running.


: ${ML_SERVE_HOME:?✋ The environment variable ML_SERVE_HOME is required}

PATH=${ML_SERVE_HOME}/python3/bin:${PATH}
export PATH

cd "$ML_SERVE_HOME"
if [ \! -f src/ray_start.py ]; then
    echo "‼️ src/ray_start.py is not found; is your ML_SERVE_HOME set correctly?" 1>&2
    exit -2
fi

# This should get called when supervisor interrupts us, but for some reason
# it never is. Thankfully ray_start does a stop on startup.
killit() {
    python3 src/ray_stop.py
    exit 0
}
trap killit 1 2 3 6 15

# Start up
python3 src/ray_start.py </dev/null
rc=$?

if [ $rc -ne 0 ]; then
    echo "🤒 ray_start failed with $rc; exiting" 1>&2
    exit -1
else
    # ray_start.py exits whether successful or not, but we want to stay in the
    # foreground because that's what supervisord expects, so start spinning.
    while :; do
        sleep 999999
    done
fi
