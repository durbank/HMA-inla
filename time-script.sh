# Script to run gsa-viz and time the total runtime

echo "Script to run and time gsa-viz"
start_time=$SECONDS

echo "Procesing gsa-viz"
python scripts/gsa-viz.py
# sleep 5
echo "Done!"

echo "Elapsed time:"
elapsed=$(( SECONDS - start_time ))
echo $elapsed
