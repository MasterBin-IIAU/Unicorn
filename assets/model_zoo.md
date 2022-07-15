# Unicorn Model Zoo
Here we provide the performance of Unicorn on multiple tasks (Object Detection, Instance Segmentation, and Object Tracking).
The complete model weights and the corresponding training logs are given by the links.

## Object Detection
The object detector of Unicorn is pretrained and evaluated on COCO. In this step, there is no segmentation head and the network is trained only using box-level annotations.
<table>
  <tr>
    <th>Experiment
    <th>Backone</th>
    <th>Box AP</th>
    <th>Model</th>
    <th>Log</th>
  </tr>
  <tr>
    <td>unicorn_det_convnext_large_800x1280</td>
    <td>ConvNext-Large</td>
    <td>53.7</td>
    <td><a href="https://drive.google.com/file/d/1kET9m1BV9f6agv5EY0oNHinJH00PKn_Q/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1QMzcK0bnPE3fcyRLHFp0W6hJdHVfgsUS/view?usp=sharing">log</a></td>
  </tr>
  <tr>
    <td>unicorn_det_convnext_tiny_800x1280</td>
    <td>ConvNext-Tiny</td>
    <td>53.1</td>
    <td><a href="https://drive.google.com/file/d/11kLsIOp6jQEEM0ZmOvvsJW_RgjgCxuYZ/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1GezYXWtUStUf01oeDvkVOFFpJ7CVh2dk/view?usp=sharing">log</a></td>
  </tr>
  <tr>
    <td>unicorn_det_r50_800x1280</td>
    <td>ResNet-50</td>
    <td>51.7</td>
    <td><a href="https://drive.google.com/file/d/13wJ8lRrIrhixDYv7zgbQ6KEhIwQH15aQ/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1E8XJHsKj5fjGTZ9Y3hMYLJKk9tQmC2pU/view?usp=sharing">log</a></td>
  </tr>

</table>

## Instance Segmentation (Optional)
Please note that this part is optional. The training of downstream tracking tasks do not rely on this. So please feel free to skip it unless you are interested in instance segmentation on COCO. In this step, a segmentaiton head is appended to the pretrained object detector. Then parameters of the object detector are frozen and only the segmentation head is optimized. So the box AP would be the same as that in the previous stage. Here we provide the results of the model with convnext-tiny backbone.

<table>
  <tr>
    <th>Experiment
    <th>Backone</th>
    <th>Mask AP</th>
    <th>Model</th>
    <th>Log</th>
  </tr>
  <tr>
    <td>unicorn_inst_convnext_tiny_800x1280</td>
    <td>ConvNext-Tiny</td>
    <td>43.2</td>
    <td><a href="https://drive.google.com/file/d/1S7wG5dzmjyeyl6gZzgJvd9EBWGO-QU2E/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1TpdECG_Vt7zaAEAkhS_l_uFEGEVeU_L0/view?usp=sharing">log</a></td>
  </tr>

</table>


## Object Tracking
There are some inner conflicts among existing MOT benchmarks. 
- Different benchmarks focus on different object classes. For example, MOT Challenge, BDD100K, and TAO include 1, 8, and 800+ object classes.
- Different benchmarks have different labeling rules. For example, the MOT challenge always annotates the whole person, even when the person is heavily occluded or cut by the image boundary. However, the other benchmarks do not share the same rule. 

These factors make it difficult to train one unified model for different MOT benchmarks. To deal with this problem, Unicorn trains two unified models. To be specific, the first model can simultaneously deal with SOT, BDD100K, VOS, and BDD100K MOTS. The second model can simultaneously deal with SOT, MOT17, VOS, and MOTS Challenge. The results of SOT and VOS are reported using the first model.

The results of the first group of models are shown as below.
<table>
  <tr>
    <th>Experiment</th>
    <th>Input Size</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>BDD100K<br>mMOTA (%)</th>
    <th>DAVIS17<br>J&F (%)</th>
    <th>BDD100K MOTS<br>mMOTSA (%)</th>
    <th>Model</th>
    <th>Log<br>Stage1</th>
    <th>Log<br>Stage2</th>
  </tr>
  <tr>
    <td>unicorn_track_large_mask</td>
    <td>800x1280</td>
    <td>68.5</td>
    <td>41.2</td>
    <td>69.2</td>
    <td>29.6</td>
    <td><a href="https://drive.google.com/file/d/1P4__Xd1wvET5Sow21_zmOx3lupAuEWN6/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1GZwqWsMgx8H3VYPZcDwk_4XxTSJxEyEf/view?usp=sharing">log1</a></td>
    <td><a href="https://drive.google.com/file/d/1eWLNiOyKFX8Tu0Xfp2whR1g7n8CBEMQ7/view?usp=sharing">log2</a></td>
  </tr>
  <tr>
    <td>unicorn_track_tiny_mask</td>
    <td>800x1280</td>
    <td>67.7</td>
    <td>39.9</td>
    <td>68.0</td>
    <td>29.7</td>
    <td><a href="https://drive.google.com/file/d/1FXDRz-9s426FRvjqZ-ghVCfv7slkyR20/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1BQPi5e_iOCQBKYj55U0Um7Y2NI69_R5z/view?usp=sharing">log1</a></td>
    <td><a href="https://drive.google.com/file/d/1dgTiATiVFyZT4xYkvxHO6kgjNSCzfd5m/view?usp=sharing">log2</a></td>
  </tr>
  <tr>
    <td>unicorn_track_tiny_rt_mask</td>
    <td>640x1024</td>
    <td>67.1</td>
    <td>37.5</td>
    <td>66.8</td>
    <td>26.2</td>
    <td><a href="https://drive.google.com/file/d/16mf7Fhs3KXY75WfX4WH8a7Sm4Dh0vjrW/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1ObMKqOr46AKmAcIC6-pTez6s0mxgTAqy/view?usp=sharing">log1</a></td>
    <td><a href="https://drive.google.com/file/d/1HdRj5ME157hDO84k6lxnA6gz1Tbe5EdQ/view?usp=sharing">log2</a></td>
  </tr>
  <tr>
    <td>unicorn_track_r50_mask</td>
    <td>800x1280</td>
    <td>65.3</td>
    <td>35.1</td>
    <td>66.2</td>
    <td>30.8</td>
    <td><a href="https://drive.google.com/file/d/1sHgysPI3-8O3U6K2ExljW4z9JrvuElSJ/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/13tDFEjFbYYZAYvDYkOXoKThDyfRT7-53/view?usp=sharing">log1</a></td>
    <td><a href="https://drive.google.com/file/d/1Qh45-TW4Nw9qx7Gkk4L6uDKGXomnnuKy/view?usp=sharing">log2</a></td>
  </tr>

</table>

The results of the second group of models are shown as below.
<table>
  <tr>
    <th>Experiment</th>
    <th>Input Size</th>
    <th>MOT17<br>MOTA (%)</th>
    <th>MOTS<br>sMOTSA (%)</th>
    <th>Model</th>
    <th>Log<br>Stage1</th>
    <th>Log<br>Stage2</th>
  </tr>
  <tr>
    <td>unicorn_track_large_mot_challenge_mask</td>
    <td>800x1280</td>
    <td>77.2</td>
    <td>65.3</td>
    <td><a href="https://drive.google.com/file/d/1tktJbsdA3peX9i8tAcDGdxwxMit0rPs0/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1NFcEkOarlhLI6jxoibKwWVqJ9jj6NE-L/view?usp=sharing">log1</a></td>
    <td><a href="https://drive.google.com/file/d/18R_IUi8ooq4ZKah0DV0ajY7Y1GO5DYvB/view?usp=sharing">log2</a></td>
  </tr>

</table>

We also provide task-specific models for users who are only interested in part of tasks.
<table>
  <tr>
    <th>Experiment</th>
    <th>Input Size</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>BDD100K<br>mMOTA (%)</th>
    <th>DAVIS17<br>J&F (%)</th>
    <th>BDD100K MOTS<br>mMOTSA (%)</th>
    <th>Model</th>
    <th>Log<br>Stage1</th>
    <th>Log<br>Stage2</th>
  </tr>
  <tr>
    <td>unicorn_track_tiny_sot_only</td>
    <td>800x1280</td>
    <td>67.5</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1NcMsWML-1-zr0SWXUOiRPNZRs-VnAWG5/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1TiKopPT93v6JDYrrKMihlBV44h6VugqS/view?usp=sharing">log1</a></td>
    <td>-</a></td>
  </tr>
  <tr>
    <td>unicorn_track_tiny_mot_only</td>
    <td>800x1280</td>
    <td>-</td>
    <td>39.6</td>
    <td>-</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1T0DxX-d_qeHvVlZ7IIbdNqtsHCQANKlQ/view?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1Fx8jataBKFH2c-q1uNVEFomQgm9252QX/view?usp=sharing">log1</a></td>
    <td>-</a></td>
  </tr>
  <tr>
    <td>unicorn_track_tiny_vos_only</td>
    <td>800x1280</td>
    <td>-</td>
    <td>-</td>
    <td>68.4</td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/12T7XodWFuwSFAv5oBcDTIBJT4INVpQ94/view?usp=sharing">model</a></td>
    <td>-</td>
    <td><a href="https://drive.google.com/file/d/1jxbAZEVgvD2pZko9jce6pJYnzYP866PQ/view?usp=sharing">log2</a></td>
  </tr>
  <tr>
    <td>unicorn_track_tiny_mots_only</td>
    <td>800x1280</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>28.1</td>
    <td><a href="https://drive.google.com/file/d/13D0rH3i0n5d_W8ead41zFySDwXAePxgv/view?usp=sharing">model</a></td>
    <td>-</a></td>
    <td><a href="https://drive.google.com/file/d/1P0GacRmGLwFw72rpaorBZ3vI63S70IeS/view?usp=sharing">log2</a></td>
  </tr>

</table>

## Structure
The downloaded checkpoints should be organized in the following structure
   ```
   ${UNICORN_ROOT}
    -- Unicorn_outputs
        -- unicorn_det_convnext_large_800x1280
            -- best_ckpt.pth
        -- unicorn_det_convnext_tiny_800x1280
            -- best_ckpt.pth
        -- unicorn_det_r50_800x1280
            -- best_ckpt.pth
        -- unicorn_track_large_mask
            -- latest_ckpt.pth
        -- unicorn_track_tiny_mask
            -- latest_ckpt.pth
        -- unicorn_track_r50_mask
            -- latest_ckpt.pth
        ...
   ```
