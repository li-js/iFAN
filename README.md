# iFAN
Integrated Face Analytics Network 

As this code was written using an old version of tensorflow and Keras, a docker is used.

To run the code, build the docker image first by
```
cd {iFAN_root}/docker
./build_docker.sh
cd {iFAN_root}/
```

Then start a docker container by 
```
./start_docker.sh
```

Within the docker container, the {iFAN_root} was mapped to /ifan/.

To run the model which performs face segmentation and landmark localization, go to folder task_seg_pts and run predict.py from there:
```
cd /ifan/task_seg_pts/
python predict.py
```

To run the model which performs face segmentation and landmark localization and emotion recognition, go to folder task_seg_pts_emo and run predict.py from there:
```
cd /ifan/task_seg_pts_emo/
python predict.py
```


Acknowledgement
Some of the codes to build the Fully Covolutionaly DenseNet are adopted from 
https://github.com/titu1994/Fully-Connected-DenseNets-Semantic-Segmentation
