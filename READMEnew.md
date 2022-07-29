# Data Pipeline
1. Generate filterd data (remove outliers and static points) and save the validation set and the test set in .pickel 
```
export PYTHONPATH=.
python src/radar_scenes_dataset_generator.py
```
2. Run segmentor on the validation set to get segmented instances and saved in .npy 
```
python src/run_instance_segmentation_for_radar_scenes.py
```
3. Tune the tracker on the validation set
```
python src/sort_instance.py
```
4. Run the tracker to associate instances in the test set and display tracks
5. Evaluate