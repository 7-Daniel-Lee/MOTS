# Data Pipeline
1. Generate filterd data (remove outliers and static points) and save the validation set and the test set in `validation_data_without_static.pickle`, `validation_label_without_static.pickle`, `test_data_without_static.pickle`, `test_label_without_static.pickle`
```
export PYTHONPATH=.
python src/radar_scenes_dataset_generator.py
```
Note: if you want fiiltered data for only one sequence, modify `data_short/sequences.json`
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