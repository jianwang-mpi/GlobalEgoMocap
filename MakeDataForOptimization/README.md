## Get Data for Optimization

1. run Mo2Cap2 or other pose estimation network to get heatmap
and joint depth files
   
2. get gt pose from BVH file and save it in to the pickle file

    see directory ```bvh_reader```

3. run OpenVSLAM on image sequence to get slam file 
   ```frame_trajectory.txt```

4. generate data for optimization

    see ```process_test_data.py```