import os


if __name__ == "__main__":
    exp_name = "unicorn_track_large_mot_challenge"
    src_dir = "Unicorn_outputs/%s/track_results_train"
    des_dir = "Unicorn_outputs/%s/track_results_dti"
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    file_list = os.listdir(src_dir)
    for f in file_list:
        src_path = os.path.join(src_dir, f)
        assert ("FRCNN" in src_path)
        des_path1 = src_path.replace(src_dir, des_dir)
        des_path2 = des_path1.replace("FRCNN", "DPM")
        des_path3 = des_path1.replace("FRCNN", "SDP")
        os.system("cp %s %s" %(src_path, des_path1))
        os.system("cp %s %s" %(src_path, des_path2))
        os.system("cp %s %s" %(src_path, des_path3))
