# from turtle import widthnp
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_Form
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox
import pandas as pd
from scipy.signal import find_peaks
import subprocess
from PIL import Image
import base64
from io import BytesIO
from pic2str import CCP_B, CCP_L, CCP_R


class Form_controller(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):

        self.ui.pushButton.clicked.connect(self.Load_R_Video)
        self.ui.pushButton_2.clicked.connect(self.Load_L_Video)
        self.ui.pushButton_3.clicked.connect(self.Load_B_Video)
        self.ui.comboBox.currentIndexChanged.connect(self.combolbox)
        self.ui.pushButton_4.clicked.connect(self.check_video)
        self.ui.pushButton_5.clicked.connect(self.begin_opticalflow)
        self.ui.pushButton_6.clicked.connect(self.crop_video)
        self.ui.pushButton_11.clicked.connect(self.Sync_video)
        self.ui.pushButton_8.clicked.connect(self.show_result)
        self.ui.pushButton_7.clicked.connect(
            self.run_code)
        self.ui.pushButton_9.clicked.connect(self.show_animat)

    def rotate(self, frame, width, height):
        if self.mode == 0:
            frame = np.rot90(frame)
            frame = np.rot90(frame)
            return frame
        elif self.mode == 1:
            return frame
        else:

            frame = np.rot90(frame)
            frame = np.rot90(frame)
            frame = np.rot90(frame)
            dim = (int(height), int(width))
            frame = cv2.resize(frame, dim,  interpolation=cv2.INTER_AREA)
            return frame

    def cutting_part(self, interval, startframe, name, video):
        cap = cv2.VideoCapture(video)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        startframe = startframe + interval
        cap.set(1, startframe)
        framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        success, frame = cap.read()
        frame = self.rotate(frame, self.width, self.height)
        box = cv2.selectROI("Select region of camera", frame)
        p1, p2 = (box[0], box[1]), (box[0] + box[2], box[1]+box[3])
        print("p1=", p1)
        print("p2=", p2)
        print("box=", box)
        cv2.destroyAllWindows()
        if name != "CCP":
            frame_cnt = 0
            red, blue = (0, 0, 255), (255, 0, 0)
            sync = 0
            global check
            global right
            while cap.isOpened() and sync == 0:
                success, frame = cap.read()
                if not success:
                    break
                frame = self.rotate(frame, self.width, self.height)
                crop_box = frame[p1[1]:p2[1], p1[0]:p2[0]]

                hsv = cv2.cvtColor(crop_box, cv2.COLOR_BGR2HSV)
                lower_white = np.array([0, 0, 0], dtype=np.uint8)

                # if glove triggered the sync , try to change upper_white or just use video_cut----------------------------------
                """if not check is 1:
                    upper_white = np.array([threshold, 30,30], dtype=np.uint8)
                else:
                    upper_white = np.array(
                        [threshold, threshold,threshold], dtype=np.uint8)
                """
                upper_white = np.array([120, 120, 120], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_white, upper_white)
                res = cv2.bitwise_and(crop_box, crop_box, mask=mask)
                if np.sum(res > 0) < 100:
                    color = red
                else:
                    if sync == 0:
                        sync = sync+frame_cnt
                    color = blue
                rect = cv2.rectangle(frame, p1, p2, color, 2, 1)
                cv2.putText(rect, str(startframe+frame_cnt+1),
                            (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(rect, str(np.sum(res > 0)),
                            (p2[0], p2[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                beginframe = startframe+frame_cnt
                if color == blue:
                    cv2.waitKey(3000)
                cv2.imshow("Detecting", frame)
                frame_cnt += 1
                if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                    break
            cap.set(cv2.CAP_PROP_POS_FRAMES, beginframe)
            self.pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print("Check Frame", self.pos)
            cv2.destroyAllWindows()
        else:
            """if check == 1:
                out = cv2.VideoWriter(
                    videopath+name+'.mp4',fourcc, fps, (int(height),int(width)))
            else:
                out = cv2.VideoWriter(
                    videopath+name+'.mp4',fourcc, fps, (int(width),int(height)))
            """
            beginframe = startframe+interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, beginframe)

            self.pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(self.pos)
            cv2.destroyAllWindows()
        return self.pos

    def combine_img(self, img1, img2, colorBGR=True):
        # 取得兩張圖片的寬高
        if colorBGR:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        else:
            img1 = img1
            img2 = img2

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # 確認兩張圖片的寬度相同，如果不同則調整尺寸
        if w1 != w2:
            scale_factor = w1 / w2
            img2 = cv2.resize(img2, (int(w1), int(h2 * scale_factor)))

        # 垂直拼接兩張圖片
        result = cv2.vconcat([img1, img2])

        # 顯示結果
        return result

    def display_img(self, checkimg):
        img = checkimg
        height, width, channel = img.shape
        bytesPerline = 3 * width
        qimg = QImage(img, width, height,
                      bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.label_4.setPixmap(QPixmap.fromImage(qimg))

    def display_frame(self, video):
        cap = cv2.VideoCapture(video)
        cap.set(0, 0)
        success, self.frame = cap.read()
        height, width, channel = self.frame.shape
        bytesPerline = 3 * width
        self.qimg = QImage(self.frame, width, height,
                           bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.label_4.setPixmap(QPixmap.fromImage(self.qimg))

    def Load_R_Video(self):
        self.Right_video = QFileDialog.getOpenFileName(
            self, "Select Right Video", "./")
        self.video_r = self.Right_video[0]
        self.name = self.video_r.split("/")[-1]
        self.ui.label_6.setText(self.name)
        print(self.video_r)
        self.display_frame(self.video_r)
        cv2.destroyAllWindows()
        self.mode = 0

    def Load_L_Video(self):
        self.Left_video = QFileDialog.getOpenFileName(
            self, "Select Right Video", "./")
        self.video_lf = self.Left_video[0]
        self.name = self.video_lf.split("/")[-1]
        self.ui.label_6.setText(self.name)
        print(self.video_lf)
        self.display_frame(self.video_lf)
        cv2.destroyAllWindows()
        self.mode = 1

    def Load_B_Video(self):
        self.Back_video = QFileDialog.getOpenFileName(
            self, "Select Right Video", "./")
        self.video_b = self.Back_video[0]
        self.name = self.video_b.split("/")[-1]
        self.ui.label_6.setText(self.name)
        self.display_frame(self.video_b)
        print(self.video_b)
        cv2.destroyAllWindows()
        self.mode = 2

    def combolbox(self):
        if self.mode == 0:
            video = self.video_r
        elif self.mode == 1:
            video = self.video_lf
        else:
            video = self.video_b
        self.text = self.ui.comboBox.currentText()
        if self.text == "Synchronization":
            self.cutting_part(0, 0, "SY", video)
            print("Start frame: Synchronization= ", self.pos)

        elif self.text == "Finger Tapping":
            self.cutting_part(25000+10500, 0,  "FT", video)  # (25000+10500
            print("Start frame: Finger tapping= ", self.pos)
            self.fingertapping_frame = 3600
        elif self.text == "Hand Movement":
            QMessageBox.question(self, 'ERROR!!!', 'DEVELOP')
        elif self.text == "Pose Trembment":
            self.cutting_part(81000+6000, 0, "PT", video)
            print("Start frame: Pose Trembment= ", self.pos)

        elif self.text == "Rest Trembment":
            QMessageBox.question(self, 'ERROR!!!', 'DEVELOP')
        else:
            QMessageBox.question(self, 'Notice!!!', 'Please select video part')

    def check_video(self):
        if self.mode == 0:
            video = self.video_r
        elif self.mode == 1:
            video = self.video_lf
        else:
            video = self.video_b
        # is the camera at back part? true=1,false=0
        global check
        check = 0
        # is the camera at right part? true=1,false=0
        global right
        right = 0
        startframe = self.pos
        cap = cv2.VideoCapture(video)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        key = 0
        cap.set(1, startframe)
        success, frame = cap.read()
        frame = self.rotate(frame, self.width, self.height)
        print("Press a -1 frame ")
        print("Press q -20 frame ")
        print("Press d +1 frame ")
        print("Press e +20 frame ")
        while(1):
            cv2.imshow(
                'Tiny adjustment for true frame we want, if dont need , press enter again', frame)
            key = cv2.waitKey(1)
            if(key == 27 or key == 13):
                break
            elif(key == 97 or key == 65):
                startframe = startframe-1
                cap.set(1, startframe)
                # if check == 1:
                #     #frame = np.rot90(frame)
                #     #frame = np.rot90(frame)
                #     #frame = np.rot90(frame)
                #     dim = (int(width*0.8), int(height*0.8))
                #     frame = cv2.resize(
                #         frame, dim, interpolation=cv2.INTER_AREA)
                # if right == 1:
                #     frame = np.rot90(frame)
                #     frame = np.rot90(frame)
                success, frame = cap.read()
                frame = self.rotate(frame, self.width, self.height)
                self.display_img(frame)
            elif(key == 100 or key == 68):
                startframe = startframe+1
                cap.set(1, startframe)
                # if check == 1:
                #     #frame = np.rot90(frame)
                #     #frame = np.rot90(frame)
                #     #frame = np.rot90(frame)
                #     dim = (int(width*0.8), int(height*0.8))
                #     frame = cv2.resize(
                #         frame, dim, interpolation=cv2.INTER_AREA)
                # if right == 1:
                #     frame = np.rot90(frame)
                #     frame = np.rot90(frame)
                success, frame = cap.read()
                frame = self.rotate(frame, self.width, self.height)
                self.display_img(frame)
            elif(key == 81 or key == 113):
                startframe = startframe-20
                cap.set(1, startframe)
                # if check == 1:
                #     #frame = np.rot90(frame)
                #     #frame = np.rot90(frame)
                #     #frame = np.rot90(frame)
                #     dim = (int(width*0.8), int(height*0.8))
                #     frame = cv2.resize(
                #         frame, dim, interpolation=cv2.INTER_AREA)
                # if right == 1:
                #     frame = np.rot90(frame)
                #     frame = np.rot90(frame)
                success, frame = cap.read()
                frame = self.rotate(frame, self.width, self.height)
                self.display_img(frame)
            elif(key == 69 or key == 101):
                startframe = startframe+20
                cap.set(1, startframe)
                # if check == 1:
                #     #frame = np.rot90(frame)
                #     #frame = np.rot90(frame)
                #     #frame = np.rot90(frame)
                #     dim = (int(width*0.8), int(height*0.8))
                #     frame = cv2.resize(
                #         frame, dim, interpolation=cv2.INTER_AREA)
                # if right == 1:
                #     frame = np.rot90(frame)
                #     frame = np.rot90(frame)
                success, frame = cap.read()
                frame = self.rotate(frame, self.width, self.height)
                self.display_img(frame)
        cv2.destroyAllWindows()

        self.finalpos = cap.get(cv2.CAP_PROP_POS_FRAMES)-1
        print(self.finalpos)

    def Sync_video(self):
        if self.mode == 0:
            video = self.video_r
            cap = cv2.VideoCapture(video)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            key = 0
            cap.set(1, self.finalpos)
            success, frame = cap.read()
            self.ckimg = frame
            check_frame = self.combine_img(
                self.ckimg1, self.ckimg, colorBGR=False)
            self.display_img(check_frame)
            cv2.imshow("Sync frame", frame)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        elif self.mode == 1:
            video = self.video_lf
            cap = cv2.VideoCapture(video)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            key = 0
            cap.set(1, self.finalpos)
            success, frame = cap.read()
            self.ckimg1 = frame
            cv2.imshow("Sync frame", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        else:
            video = self.video_b
            cap = cv2.VideoCapture(video)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            key = 0
            cap.set(1, self.finalpos)
            success, frame = cap.read()
            frame = np.rot90(frame)
            frame = np.rot90(frame)
            frame = np.rot90(frame)
            dim = (int(height*0.8), int(width*0.8))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            self.ckimg2 = frame
            check_frame = self.combine_img(
                self.ckimg1, self.ckimg2, colorBGR=False)
            self.display_img(check_frame)
            cv2.imshow("Sync frame", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

    def crop_video(self):
        creat_path = "result"
        if self.mode == 0:
            video = self.video_r
        elif self.mode == 1:
            video = self.video_lf
        else:
            video = self.video_b
        cap = cv2.VideoCapture(video)
        cap.set(1, self.finalpos)
        _, frm = cap.read()
        frm = self.rotate(frm, self.width, self.height)
        self.rect = cv2.selectROI("ROI", frm, True, False)
        self.crop_gray = np.array(
            frm[self.rect[1]:self.rect[1]+self.rect[3], self.rect[0]:self.rect[0]+self.rect[2], :])
        cv2.destroyWindow('ROI')
        print(self.rect)
        if not os.path.isdir(creat_path):
            os.mkdir(creat_path)

    def begin_opticalflow(self):
        if self.mode == 0:
            video = self.video_r
        elif self.mode == 1:
            video = self.video_lf
        else:
            video = self.video_b
        k = -1
        dots = []   # 記錄座標的空串列
        self.filename_prefix = video.split("_")[0]

        def onMouse(event, x, y, flags, param):
            if event == 1:
                dots.append([x, y])                          # 記錄座標
                cv2.circle(self.crop_gray, (x, y), 3,
                           (0, 0, 255), -1)   # 在點擊的位置，繪製圓形
                num = len(dots)
                # 目前有幾個座標
                print(f"Chose point {num}")
                if num > 1:                                  # 如果有兩個點以上
                    x1 = dots[num-2][0]
                    y1 = dots[num-2][1]
                    x2 = dots[num-1][0]
                    y2 = dots[num-1][1]
                    cv2.line(self.crop_gray, (x1, y1), (x2, y2),
                             (0, 0, 255), 2)  # 取得最後的兩個座標，繪製直線
                cv2.imshow('Please select point', self.crop_gray)

        cv2.namedWindow("Please select point")
        cv2.setMouseCallback("Please select point", onMouse)
        cap = cv2.VideoCapture(video)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.finalpos)
        while True:
            _, frm = cap.read()
            frm = self.rotate(frm, self.width, self.height)
            cv2.imshow("Please select point", self.crop_gray)
            if cv2.waitKey(50000) == 27 or k == -1:
                old_gray = cv2.cvtColor(self.crop_gray, cv2.COLOR_BGR2LAB)
                cv2.destroyAllWindows()
                break
        print(f"The center is {dots}")

        old_pts = np.array(dots, dtype="float32")
        mask = np.zeros_like(self.crop_gray)

        prev_frame = None
        lk_params = dict(winSize=(15, 15), maxLevel=4,  criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.001))
        all_pts = []
        self.tracknumber = len(dots)
        for i in range(self.tracknumber):
            all_pts.append([])
        frame_count = 0
        while True:
            ret, frame2 = cap.read()

            if ret == True:
                frame_count += 1
                print("Frame: ", frame_count)
                frame2 = self.rotate(frame2, self.width, self.height)
                crop_frame2 = np.array(frame2[self.rect[1]:self.rect[1]+self.rect[3],
                                              self.rect[0]:self.rect[0]+self.rect[2], :])
                new_gray = cv2.cvtColor(crop_frame2, cv2.COLOR_BGR2LAB)

                if prev_frame is None:
                    prev_frame = new_gray
                    continue

                new_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, new_gray, old_pts, None, **lk_params)

                good_pts = old_pts
                good_next_pts = new_pts

                for i in range(10):
                    # Calculate the forward-backward error
                    fb_error = good_pts - \
                        cv2.calcOpticalFlowPyrLK(
                            new_gray, old_gray, good_next_pts, None, **lk_params)[0]
                    fb_error_total = fb_error.copy()
                    for j in range(10):
                        rev_pts, status, err = cv2.calcOpticalFlowPyrLK(
                            new_gray, old_gray, good_next_pts + fb_error, None, **lk_params)
                        fb_error_rev = good_pts - rev_pts
                        fb_error = (fb_error + fb_error_rev) / 2
                        fb_error_total += fb_error
                    fb_error_total /= 100
                    good_next_pts += fb_error_total

                # Apply median filtering to smooth the optical flow field
                x_flow = good_next_pts[0][..., 0]
                y_flow = good_next_pts[0][..., 1]
                x_flow = cv2.medianBlur(x_flow, 5)
                y_flow = cv2.medianBlur(y_flow, 5)
                good_next_pts[0][..., 0] = x_flow
                good_next_pts[0][..., 1] = y_flow

                old_gray = new_gray.copy()
                old_pts = good_next_pts.copy()

                for i in range(len(new_pts)):

                    cv2.circle(crop_frame2, (int(good_next_pts[i][0]), int(
                        good_next_pts[i][1])), 2, (255, 0, 0), 2)  # check point
                    cv2.rectangle(crop_frame2, (int(good_next_pts[i][0])-10, int(good_next_pts[i][1])-10), (int(
                        good_next_pts[i][0])+10, int(good_next_pts[i][1])+10), (0, 0, 255), 2)
                    cv2.putText(crop_frame2, "Point "+str(i+1), (int(good_next_pts[i][0]-10), int(
                        good_next_pts[i][1]-10)), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)

                    cv2.circle(mask, (int(good_next_pts[i][0]), int(
                        good_next_pts[i][1])), 2, (0, 255, 0), 2)
                    combined = cv2.addWeighted(
                        crop_frame2, 0.7, mask, 0.3, 0.1)
                    all_pts[i].append(good_next_pts[i])
                # print(good_next_pts.shape)
                print(good_next_pts)
                # cv2.imshow("chech area",crop_frame2)
                cv2.imshow("The Area", mask)
                cv2.imshow("windows", combined)
                # 3600
                if self.text == "Finger Tapping" and cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.finalpos + 3600:
                    cv2.waitKey(2000)  # Wait for 2 seconds
                    break
                # 4800
                elif self.text == "Pose Trembment" and cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.finalpos + 1000:
                    cv2.waitKey(2000)  # Wait for 2 seconds
                    break
                elif cv2.waitKey(1) == 27:  # Wait for ESC key
                    break
            else:
                break
        for i in range(self.tracknumber):
            filename_prefix = self.name.split("_")[0]
            output_path = f"result/{filename_prefix}"
            result_path = f"result/{filename_prefix}/{self.rect[0]}_{self.rect[1]}_crop_Point{i+1}_{dots[i][0]}_{dots[i][1]}_preds_"
            if self.mode == 0:
                result_path += "L_R.npy"
            elif self.mode == 1:
                result_path += "L_L.npy"
            else:
                result_path += "L_B.npy"

            if not os.path.isdir(output_path):
                os.mkdir(output_path)
            np.save(result_path, all_pts[i])
        cv2.destroyAllWindows()
        cap.release()
        QMessageBox.question(
            self, 'Notice!!!', f"Store in {video} ")

    def calculate_3D_point(self, mtx1, dist1, mtx2, dist2, R1, R2, P1, P2, leftpoint, rightpoint):
        projPoints1 = np.array(leftpoint).reshape(-1, 1, 2)
        projPoints2 = np.array(rightpoint).reshape(-1, 1, 2)

        # Undistort the 2D points
        projPoints1 = cv2.undistortPoints(
            src=projPoints1, cameraMatrix=mtx1, distCoeffs=dist1, R=R1, P=P1)
        projPoints2 = cv2.undistortPoints(
            src=projPoints2, cameraMatrix=mtx2, distCoeffs=dist2, R=R2, P=P2)

        # Triangulate the 2D points to get corresponding 3D points (actually 4D because using homogeneous coordinates)
        points4d = []
        projPoints1, projPoints2 = projPoints1.reshape(
            -1, 2).T, projPoints2.reshape(-1, 2).T
        points4d.append(cv2.triangulatePoints(
            P1, P2, projPoints1, projPoints2))

        points4d = np.array(points4d)
        # np.save("./points4d.npy", points4d)

        # Transform points from homogeneous coordinates to cartesian coordinate system (4D to 3D)
        points3d = []
        for i in range(points4d.shape[0]):
            points3d.append(np.array([points4d[i, :, j]
                                      for j in range(points4d.shape[-1])]))
        pi = []
        for each_points3d in points3d:
            pi.append(np.array([each_points3d[i][:-1] / each_points3d[i][-1]
                                for i in range(each_points3d.shape[0])]))

        # Save results
        dist = []
        for i in range(len(pi[i]) // 2):
            dist.append(np.linalg.norm(pi[0][i*2]-pi[0][i*2+1]) * 10)

        """# plt.hist(dist, bins=(131-94)*10, range=(94, 131))
        #plt.hist(dist, bins=int((max(dist)-min(dist))*10),range=(min(dist), max(dist)))
        plt.title("Points Distances Distribution Over 240 Frames with 4 Chessboard")
        plt.xlabel("Distance (mm)")
        plt.ylabel("Counts")
        plt.savefig("points_distances_dist.png")
        plt.clf()

        plt.plot(dist)
        plt.title(f"Points Distances Over 240 Frames with 4 Chessboard")
        plt.xlabel("Frame Count")
        plt.ylabel("Distance (mm)")
        plt.savefig("points_distances.png")
        plt.clf()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        res = stats.probplot(dist, dist="norm", plot=ax)
        ax.set_title(f"Normal Probability Plot")
        plt.savefig("normal_prob_plot.png")
        plt.show()"""

        return pi

    def load_cam_info(self, cam_pair):  # input "B_L" or "B_R" or "R_L"
        mtx1 = np.load(
            f"camera_calibration/calibrate_info/{cam_pair}/mtx1.npy")
        dist1 = np.load(
            f"camera_calibration/calibrate_info/{cam_pair}/dist1.npy")
        mtx2 = np.load(
            f"camera_calibration/calibrate_info/{cam_pair}/mtx.npy")
        dist2 = np.load(
            f"camera_calibration/calibrate_info/{cam_pair}/dist2.npy")
        R1 = np.load(
            f"camera_calibration/calibrate_info/{cam_pair}/R1.npy")
        R2 = np.load(
            f"camera_calibration/calibrate_info/{cam_pair}/R2.npy")
        P1 = np.load(
            f"camera_calibration/calibrate_info/{cam_pair}/P1.npy")
        P2 = np.load(
            f"camera_calibration/calibrate_info/{cam_pair}/P2.npy")
        return mtx1, dist1, mtx2, dist2, R1, R2, P1, P2

    def retify(self, point, coor):
        cx = 1280/2
        cy = 720/2
        temp = point+coor
        old_coor_offset = [temp[0]-cx, temp[1]-cy]
        return [cx + old_coor_offset[0], cy + old_coor_offset[1]]

    def check_coor(self, frame, coor, target, back=False):
        # cap = cv2.VideoCapture(frame)
        # cap.set(1, 39300.0)
        # _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if back:
            frame = self.rotate(frame, 1280, 720)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            coor = self.retify(target, coor)
            print("B", coor)
        else:
            coor = [coor[0] + target[0], coor[1] + target[1]]
            print("L", coor)
        cv2.circle(frame, (int(coor[0]), int(coor[1])), 10, (0, 255, 0), 10)
        # plt.imshow(img)
        # plt.show()
        return frame

    def detect_chessboard_L(self, img, number):
        # cb_number = str(cb_number)
        mask_num = 4
        """
        # Flip the image for right view to get the correct order
        if pos == "right":
            frame = cv2.flip(frame, -1)
        """
        # Select the chessboards
        bboxes = []
        while len(bboxes) < mask_num:
            bbox = cv2.selectROI("Select chessboards", img)
            p1, p2 = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
            bboxes.append([p1, p2])
            # print(bboxes)
            # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                break
            cv2.rectangle(img, p1, p2, (0, 0, 0), -1)
            # plt.imshow(img)
            # plt.show()

        cbrow, cbcol = 8, 6  # chessboard for grid
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (cbrow, cbcol), corners2, ret)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("img", img)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
            # filename_chessboard = f"{number}_Leftside_corner.npy"
            # np.save(filename_chessboard, corners2, allow_pickle=True)
            # df = pd.DataFrame()
            # df['px'] = corners.T[0][0]
            # df['py'] = corners.T[1][0]
            # df.to_csv(f"{number}_Leftside_corner.csv", index=False)
            return corners2

        else:
            print(f"Image fail to detect")
        print("Go through all imgs, can't detect corner")
        cv2.destroyAllWindows()

    def detect_chessboard_R(self, inputimg, number):
        # cb_number = str(cb_number)
        img = cv2.rotate(inputimg, cv2.ROTATE_180)
        mask_num = 4
        """
        # Flip the image for right view to get the correct order
        if pos == "right":
            frame = cv2.flip(frame, -1)
        """
        # Select the chessboards
        bboxes = []
        while len(bboxes) < mask_num:
            bbox = cv2.selectROI("Select chessboards", img)
            p1, p2 = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
            bboxes.append([p1, p2])
            # print(bboxes)
            # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                break
            cv2.rectangle(img, p1, p2, (0, 0, 0), -1)

        # plt.imshow(img)
        # plt.show()

        cbrow, cbcol = 8, 6  # chessboard for grid
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (cbrow, cbcol), corners2, ret)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("img", img)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
            # filename_chessboard = f"{number}_Rightside_corner.npy"
            # np.save(filename_chessboard, corners2, allow_pickle=True)
            # df = pd.DataFrame()
            # df['px'] = corners.T[0][0]
            # df['py'] = corners.T[1][0]
            # df.to_csv(f"{number}_Rightside_corner.csv", index=False)
            return corners2
        else:
            print(f"Image fail to detect")
        print("Go through all imgs, can't detect corner")
        cv2.destroyAllWindows()

    def detect_chessboard_B(self, img, number):

        mask_num = 4
        """
        # Flip the image for right view to get the correct order
        if pos == "right":
            frame = cv2.flip(frame, -1)
        """
        # Select the chessboards
        bboxes = []
        while len(bboxes) < mask_num:
            bbox = cv2.selectROI("Select chessboards", img)
            p1, p2 = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
            bboxes.append([p1, p2])
            # print(bboxes)
            # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            print("Press any other key to select next object")
            k = cv2.waitKey(0) & 0xFF
            if (k == 113):  # q is pressed
                break
            cv2.rectangle(img, p1, p2, (0, 0, 0), -1)

        # plt.imshow(img)
        # plt.show()

        cbrow, cbcol = 8, 6  # chessboard for grid
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (cbrow, cbcol), corners2, ret)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("img", img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            # filename_chessboard = f"{number}_Backside_corner.npy"
            # np.save(filename_chessboard, corners2, allow_pickle=True)
            # df = pd.DataFrame()
            # df['px'] = corners.T[0][0]
            # df['py'] = corners.T[1][0]
            # df.to_csv(f"{number}_Backside_corner.csv", index=False)
            return corners2
        else:
            print(f"Image fail to detect")
        print("Go through all imgs, can't detect corner")
        cv2.destroyAllWindows()

    def unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    # input 3d points of chessboard
    def get_transformation_matrix(self, point3d):
        # target coordinate system
        chessboardpoints = [
            [point3d[40][0], point3d[40][1], point3d[40][2]],
            [point3d[41][0], point3d[41][1], point3d[41][2]],
            [point3d[32][0], point3d[32][1], point3d[32][2]]
        ]
        cp0, cp1, cp2 = chessboardpoints
        cx0, cy0, cz0 = cp0

        cx1, cy1, cz1 = cp1
        cx2, cy2, cz2 = cp2

        cux, cuy, cuz = cu = [cx1-cx0, cy1-cy0, cz1-cz0]  # Vector c0->c1
        normalized_cu = cu/np.linalg.norm(cu)
        cvx, cvy, cvz = cv = [cx2-cx0, cy2-cy0, cz2-cz0]  # Vector c0->c2
        normalized_cv = cv/np.linalg.norm(cv)
        c_u_cross_v = np.cross(cu, cv)  # Normal vector
        # c_u_cross_v = [cuy*cvz - cuz*cvy, cuz*cvx - cux*cvz, cux*cvy - cuy*cvx] #Normal vector
        c_a, c_b, c_c = c_u_cross_v
        normalized_cuv = c_u_cross_v/np.linalg.norm(c_u_cross_v)
        n_cvx, n_cvy, n_cvz = new_cv = np.cross(cu, c_u_cross_v)
        normalized_cv = new_cv/np.linalg.norm(new_cv)
        # original coordinate system
        original_system = [[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]]
        m11 = np.dot(normalized_cv, original_system[0])
        m12 = np.dot(normalized_cv, original_system[1])
        m13 = np.dot(normalized_cv, original_system[2])
        #
        m21 = np.dot(normalized_cu, original_system[0])
        m22 = np.dot(normalized_cu, original_system[1])
        m23 = np.dot(normalized_cu, original_system[2])
        #
        m31 = np.dot(normalized_cuv, original_system[0])
        m32 = np.dot(normalized_cuv, original_system[1])
        m33 = np.dot(normalized_cuv, original_system[2])
        transformation_matrix = np.array(([m11, m12, m13],
                                          [m21, m22, m23],
                                          [m31, m32, m33],
                                          ))
        original_matrix = transformation_matrix
        theta = np.radians(float(270))
        R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        transformation_matrix = R_z.dot(original_matrix)
        # print("Transformation_matrix :\n", transformation_matrix)
        return transformation_matrix

    def SSA(self, series, window):
        #     series = series - np.mean(series)
        # step1 嵌入
        windowLen = int(window)             # 嵌入窗口長度
        seriesLen = len(series)     # 序列
        K = seriesLen - windowLen + 1
        X = np.zeros((windowLen, K))
        for i in range(K):
            X[:, i] = series[i:i + windowLen]

        # step2: svd分解， U和sigma已經按升序排序
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)

        for i in range(VT.shape[0]):
            VT[i, :] *= sigma[i]
        A = VT
        rec = np.zeros((windowLen, seriesLen))
        for i in range(windowLen):
            for j in range(windowLen-1):
                for m in range(j+1):
                    rec[i, j] += A[i, j-m] * U[m, i]
                rec[i, j] /= (j+1)
            for j in range(windowLen-1, seriesLen - windowLen + 1):
                for m in range(windowLen):
                    rec[i, j] += A[i, j-m] * U[m, i]
                rec[i, j] /= windowLen
            for j in range(seriesLen - windowLen + 1, seriesLen):
                for m in range(j-seriesLen+windowLen, windowLen):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= (seriesLen - j)
        return rec[0, :]

    def calculate_3d_datas_and_feature(self, information):
        '''
        This function include:
        1. calculate 3d data
        2. coordinate translation (origin)
        3. calculate features

        Input object should include the inforamtion with this order:
        1. filename_prefix 2. result_root_path
        3. in1_L filename 4. th1_L filename 5. in1_B filename 6. th1_B filename
        7. left_img_path 8. back_check_img 9. coorL 10. coorB
        '''
        show_fig = False
        creat_3d_path = "result/results_3d"
        if not os.path.isdir(creat_3d_path):
            os.mkdir(creat_3d_path)
        result_root_path = f"result/results_3d/{information[0]}/"
        if not os.path.isdir(result_root_path):
            os.mkdir(result_root_path)

        filename_prefix = information[0]
        path_for_analysis = information[1]
        points_L = {}
        points_B = {}

        # Iterate over the files in the 'information' list
        for file_path in information:
            if "_L_L" in file_path:
                if "Point" in file_path:
                    # Extract the point number from the file name
                    point_num = int(file_path.split("Point")[1].split("_")[0])
                    # Load the numpy array and store it in the dictionary of points
                    points_L[point_num] = np.load(
                        path_for_analysis + file_path)
            elif "_L_B" in file_path:
                if "Point" in file_path:
                    # Extract the point number from the file name
                    point_num = int(file_path.split("Point")[1].split("_")[0])
                    # Load the numpy array and store it in the dictionary of points
                    points_B[point_num] = np.load(
                        path_for_analysis + file_path)

        mtx1, dist1, mtx2, dist2, R1, R2, P1, P2 = self.load_cam_info(
            "B_L_corr")
        coorL = information[-2]  # L, the crop coordinate
        coorB = information[-1]  # B, the crop coordinate

        # self.ckimg1
        # ckimg2
        left_check_img = self.check_coor(
            self.ckimg1, coorL, points_L[1][0], back=False)
        back_check_img = self.check_coor(
            self.ckimg2, coorB, points_B[1][0], back=True)
        check_frame = self.combine_img(
            back_check_img, left_check_img, colorBGR=True)
        self.display_img(check_frame)
        t, arr = plt.subplots(1, 2, figsize=(12, 5))
        arr[0].imshow(left_check_img)
        arr[1].imshow(back_check_img)
        plt.savefig(result_root_path+filename_prefix +
                    "_R_hand_FLOW_checkCoor.png")
        if show_fig:
            plt.show()
        # create a list of lists to store the points
        point3d = [[] for i in range(self.tracknumber)]
        dt = []

        for i in range(len(points_L[1])):
            for j in range(self.tracknumber):
                # calculate the 3D point for each view
                point3d[j].append(self.calculate_3D_point(mtx1, dist1, mtx2, dist2, R1, R2, P1, P2,
                                                          points_B[j+1][i]+coorB, self.retify(points_L[j+1][i], coorL))[0][0])
            # append the time values to the dt list
            dt.append(i/240)

        # convert the point lists to numpy arrays
        point3d = [np.array(p) for p in point3d]
        print(point3d)
        # print(
        #     f"index length: {len(point3d[1])}, thumb length: {len(point3d[0])}")

        data = {"time": dt}

        # iterate over the point views and dynamically create the column names
        for i, p in enumerate(point3d):
            for j in "xyz":
                col_name = f"Point{i+1}_{j}"
                # select the column based on the letter
                data[col_name] = p[:, ord(j)-ord('x')]
        # calculate the distance between points 1 and 2
        dis = np.sum(np.sqrt((point3d[1] - point3d[0])**2), axis=1)

        # add the distance column to the data dictionary
        data["dis"] = dis

        # create the dataframe from the data dictionary
        df = pd.DataFrame(data)
        df.to_csv(result_root_path + filename_prefix +
                  "_R_hand_FLOW_3d_pixel.csv", index=False)

        # pixel convert to cm and coordinate transformation
        byte_data = base64.b64decode(CCP_L)
        image_data = BytesIO(byte_data)
        image = Image.open(image_data)
        img_L = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        left_ccp_img = img_L  # "camera_calibration/calibration data/CCP_img_L.png"
        # right_ccp_img = "Camera_calibrate/CCP_img_R.png"
        byte_data = base64.b64decode(CCP_B)
        image_data = BytesIO(byte_data)
        image = Image.open(image_data)
        img_B = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        back_ccp_img = img_B  # "camera_calibration/calibration data/CCP_img_B.png"

        chessboard_point_left = self.detect_chessboard_L(
            left_ccp_img, "Chessboard_1")

        # chessboard_point_right = detect_chessboard_R(right_ccp_img,"Chessboard_1")
        chessboard_point_back = self.detect_chessboard_B(
            back_ccp_img, "Chessboard_1")

        chess_corner = []
        for i in range(len(chessboard_point_left)):
            chess_corner.append(self.calculate_3D_point(mtx1, dist1, mtx2, dist2, R1, R2, P1, P2,
                                                        chessboard_point_back[i], chessboard_point_left[i])[0][0])
        # chess_corner = np.array(chess_corner)
        d = np.sqrt(np.sum((chess_corner[0] - chess_corner[1])**2))
        d = d/10
        print(f"Distance between two pint: {d}, {d} = 1mm")
        chess_corner = chess_corner/d
        point3d = [data/d for data in point3d]
        transformation_matrix = self.get_transformation_matrix(chess_corner)
        chess_corner_trans = []  # convert origin to left down chessboard
        for i in range(len(chess_corner)):  # rotate data points
            chess_corner_trans.append(
                np.dot(transformation_matrix, chess_corner[i].reshape(3, 1)).reshape(1, 3)[0])

        # create a list of lists to store the points
        point3d_trans = [[] for i in range(self.tracknumber)]
        for i in range(len(points_L[1])):
            for j in range(self.tracknumber):
                point3d_trans[j].append(np.dot(transformation_matrix,
                                        point3d[j][i].reshape(3, 1)).reshape(1, 3)[0])

        origin_point = chess_corner_trans[40]  # translate data points
        chess_corner_trans = np.array(chess_corner_trans) - origin_point
        point3d_trans = [data-origin_point for data in point3d_trans]
        # print(index_trans)
        data = {"time": dt}
        np.save(result_root_path+filename_prefix +
                "_chess_corner_trans", chess_corner_trans)
        np.save(result_root_path+filename_prefix +
                "_point3d_trans", point3d_trans)
        # iterate over the point views and dynamically create the column names
        for i, p in enumerate(point3d_trans):
            for j in "xyz":
                col_name = f"Point{i+1}_{j}"
                data[col_name] = p[:, ord(j)-ord('x')]
        dis = np.sum(np.sqrt((point3d_trans[1] - point3d_trans[0])**2), axis=1)
        data["dis"] = dis
        df = pd.DataFrame(data)
        df.to_csv(result_root_path + filename_prefix +
                  "_R_hand_FLOW_3d_mm_trans.csv", index=False)

        dis = np.array(self.SSA(dis, 5))  # 20
        norm_dis = (dis - np.min(dis))/(np.max(dis) - np.min(dis))

        peak_idx, _ = find_peaks(dis, distance=25, prominence=0.5)
        # valley_idx, _ = find_peaks(-dis, distance=25, prominence = 0.3)
        valley_idx = []
        for j in range(1, len(peak_idx)):
            p_interval = dis[peak_idx[j-1]+10:peak_idx[j]-10]
            p_derivative = self.SSA(np.diff(p_interval), 3)
            idx = np.where(abs(p_derivative) <= 0.2)
            threshold = p_interval[idx[0][0]]
            idx_t = np.where(p_interval <= threshold)
            middle_idx = peak_idx[j-1]+10 + idx_t[0][int(len(idx_t[0])/2)]
            valley_idx.append(middle_idx)

        t = np.arange(len(dis))  # distance=45,
        plt.figure(figsize=(6, 4))
        plt.plot(norm_dis)
        plt.plot(t[peak_idx], norm_dis[peak_idx], 'r.')
        plt.plot(t[valley_idx], norm_dis[valley_idx], 'b.')
        # plt.savefig(result_root_path+filename_prefix+"_R_hand_DFE_overview.png")
        if show_fig:
            plt.show()

        if abs(len(peak_idx) - len(valley_idx)) > 1:
            print("Check peak and valley again")

        tapping_times = np.min((len(peak_idx), len(valley_idx)))
        average_freq = tapping_times / 15
        QMessageBox.question(
            self, 'Finish!!', f"Subject had done {tapping_times} tappings")

        close_speed, open_speed, speed, accerate = [], [], [], []

        open_amplitude, peak_amplitude, valley_amplitude = [], [], []
        peak_period = []
        for i in range(tapping_times):
            close_speed.append(
                (norm_dis[peak_idx[i]] - norm_dis[valley_idx[i]])/(dt[valley_idx[i]]-dt[peak_idx[i]]))
            # speed.append(close_speed[-1])
            open_amplitude.append(dis[peak_idx[i]] - dis[valley_idx[i]])

            peak_amplitude.append(norm_dis[peak_idx[i]])
            valley_amplitude.append(norm_dis[valley_idx[i]])
            if i >= 1:
                open_speed.append(
                    (norm_dis[peak_idx[i]] - norm_dis[valley_idx[i-1]])/(dt[valley_idx[i-1]]-dt[peak_idx[i]]))
                # speed.append(open_speed[-1])
                # speed.append( abs((norm_dis[i] - norm_dis[i-1]))/(dt[i]-dt[i-1]) )
                peak_period.append(dt[peak_idx[i]] - dt[peak_idx[i-1]])
        for i in range(1, len(dis)):
            speed.append(abs((norm_dis[i] - norm_dis[i-1]))/(dt[i]-dt[i-1]))
        for i in range(1, len(speed)):
            accerate.append(speed[i] - speed[i-1])

        peak_y = []
        valley_y = []
        for i in range(len(peak_idx)):
            peak_y.append(dis[peak_idx[i]])
        for i in range(len(valley_idx)):
            valley_y.append(dis[valley_idx[i]])
        if show_fig:
            plt.show()

        # np.save("speed",speed, allow_pickle=True)
        # df_time = pd.DataFrame({"Time":dt})
        df_peaks = pd.DataFrame({'peaks': peak_idx})
        df_valleys = pd.DataFrame({'valleys': valley_idx})
        df_openspeed = pd.DataFrame({'open_speed': open_speed})
        df_closespeed = pd.DataFrame({'close_speed': close_speed})
        df_speed = pd.DataFrame({'speed': speed})
        df_accerate = pd.DataFrame({"accerate": accerate})
        df_OpenAmplitude = pd.DataFrame({'amplitude': open_amplitude})
        df_PeakAmplitude = pd.DataFrame({'amplitude': peak_amplitude})
        df_ValleyAmplitude = pd.DataFrame({'amplitude': valley_amplitude})
        df_peak_period = pd.DataFrame({'amplitude': peak_period})
        df_feature = pd.concat([df_peaks, df_valleys, df_openspeed, df_closespeed, df_speed, df_accerate, df_OpenAmplitude, df_PeakAmplitude, df_ValleyAmplitude, df_peak_period],
                               ignore_index=True, axis=1)
        df_feature.columns = ["peaks", "valleys", "open_speed", "close_speed",
                              "speed", "accerate", "open_amplitude", "peak_amplitude", "valley_amplitude", "peak_period"]

        df_feature.to_csv(result_root_path + filename_prefix +
                          "_R_hand_FLOW_3d_features.csv", index=False)
        plt.close("all")  # close all fig
        self.peak_y = peak_y
        self.valley_y = valley_y
        self.peak_idx = peak_idx
        self.valley_idx = valley_idx
        self.dis = dis
        self.chess_corner = chess_corner
        self.point3d = point3d
        self.chess_corner_trans = chess_corner_trans
        self.point3d_trans = point3d_trans
        self.filename_prefix = filename_prefix

    def show_animat(self,name):
        subprocess.run(["python", 'animation.py'])

    def show_result(self):
        filename_prefix = self.name.split("_")[0]
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.minorticks_on()
        # 显示副刻度线
        ax1.tick_params(axis="both", which="major",
                        direction="in", width=1, length=5)
        ax1.tick_params(axis="both", which="minor",
                        direction="in", width=1, length=3)
        plt.title("Subject "+self.name.split("_")[0]+" FTT")
        plt.xlabel("Frame")
        plt.grid(True, which="major", linestyle="--",
                 color="gray", linewidth=0.75)
        plt.grid(True, which="minor", linestyle=":",
                 color="lightgray", linewidth=0.75)
        ax2 = ax1.twinx()
        ax1.set_ylabel("Distance(mm)")
        ax1.plot(self.dis, linewidth='2', label="Distance")
        ax1.scatter(self.peak_idx, self.peak_y, label="Peak", color="red")
        ax1.scatter(self.valley_idx, self.valley_y,
                    label="Valley", color="blue")
        ax2.set_ylabel('Change in speed (mm/frame)', color='black')
        # ax2.plot(speed, linewidth='1.5', label="Speed",
        #          color='green', linestyle='--', alpha=0.7)
        # ax2.plot(accerate, linewidth='1.5', label="Accerate",
        #          color='red', linestyle='-.', alpha=0.5)
        fig.tight_layout()
        fig.legend(loc=1, bbox_to_anchor=(0.95, 0.95))

        plt.savefig(f"result/results_3d/{filename_prefix}/Feature_overview.png")
        print()
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('$Coordinate-X$', fontsize=20)
        ax.set_ylabel('$Coordinate-Y$', fontsize=20)
        ax.set_zlabel('$Coordinate-Z$', fontsize=20)

        ax.scatter(0, 0, 0)
        ax.quiver(
            0, 0, 0,  # <-- starting point of vector
            [300, 0, 0], [0, 300, 0], [0, 0, 300],  # <-- directions of vector
            color='red', lw=1,
        )
        corr_points = [40, 41, 32]
        for i in range(0, len(self.chess_corner)):
            if i in corr_points:
                ax.scatter(self.chess_corner[i][0], self.chess_corner[i][1],
                           self.chess_corner[i][2], s=60, marker="*")
            else:
                ax.scatter(
                    self.chess_corner[i][0], self.chess_corner[i][1], self.chess_corner[i][2])
        for i in range(1):
            for j in range(self.tracknumber):
                ax.scatter(self.point3d[j][i][0], self.point3d[j][i]
                           [1], self.point3d[j][i][2])
        # ax.quiver(
        #     point3d[0][0], point3d[0][1], point3d[0][2], # <-- starting point of vector
        #     5*unit_vector(point3d[1]), 5*unit_vector(point3d[8]), [0,0,3], # <-- directions of vector
        #     color = 'red', lw = 1,
        #     )
        # ax.quiver(
        #     point3d[0][0], point3d[0][1], point3d[0][2], # <-- starting point of vector
        #     3*(point3d[0][1] - point3d[0]), 3*(point3d[8] - point3d[0]), [0,0,3], # <-- directions of vector
        #     color = 'red', lw = 1,
        #     )
        ax.view_init(45, 45)
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(0, 0, 0)
        # ax.set_xlim(0,35)
        # ax.set_ylim(0,35)
        # ax.set_zlim(0,60)
        ax.set_xlabel('$Coordinate-X$', fontsize=20)
        ax.set_ylabel('$Coordinate-Y$', fontsize=20)
        ax.set_zlabel('$Coordinate-Z$', fontsize=20)
        corr_points = [40, 41, 32]
        for i in range(0, len(self.chess_corner_trans)):
            if i in corr_points:
                ax.scatter(self.chess_corner_trans[i][0], self.chess_corner_trans[i]
                           [1], self.chess_corner_trans[i][2], s=60, marker="*")
            else:
                ax.scatter(
                    self.chess_corner_trans[i][0], self.chess_corner_trans[i][1], self.chess_corner_trans[i][2])
        for i in range(1):
            for j in range(self.tracknumber):  # 47
                ax.scatter(self.point3d_trans[j][i][0], self.point3d_trans[j][i]
                           [1], self.point3d_trans[j][i][2])
                ax.text(self.point3d_trans[j][i][0], self.point3d_trans[j][i]
                        [1], self.point3d_trans[j][i][2], f"3D Point{j+1}", size=8, zorder=1)
        ax.plot([0, 0], [0, 0], [0, 300], color="green")
        ax.plot([0, 0], [0, 320], [0, 0], color="black")
        ax.plot([0, 350], [0, 0], [0, 0], color="gray")
        ax.view_init(45, 45)
        plt.show()

        # for i in range(0, len(self.chess_corner_trans)):
        #     if i in corr_points:
        #         ax.scatter(self.chess_corner_trans[i][0], self.chess_corner_trans[i]
        #                    [1], self.chess_corner_trans[i][2], s=60, marker="*")
        #     else:
        #         ax.scatter(
        #             self.chess_corner_trans[i][0], self.chess_corner_trans[i][1], self.chess_corner_trans[i][2])
        # for i in range(1):
        #     for j in range(self.tracknumber):  # 47
        #         ax.scatter(self.point3d_trans[j][i][0], self.point3d_trans[j][i]
        #                    [1], self.point3d_trans[j][i][2])
        #         ax.text(self.point3d_trans[j][i][0], self.point3d_trans[j][i]
        #                 [1], self.point3d_trans[j][i][2], f"3D Point{j+1}", size=8, zorder=1)

    def run_code(self):
        print("="*15+" Processing subject "+"="*15)
        filename_prefix = self.name.split("_")[0]
        # temp = "03555114_20221128_Hand_L_B_deflickerFT_deflicker.mp4"
        # filename_prefix = temp.split("_")[0]
        result_root_path = f"result/{filename_prefix}/"

        filenames = os.listdir(result_root_path)
        information = [filename_prefix, result_root_path]

        for filename in filenames:
            parts = filename.replace(".npy", "").split("_")
            # print(parts)
            if "preds" in parts:
                if "L" == parts[-1] and "L" in parts:
                    for i in range(self.tracknumber+1):  # self.tracknumber+1
                        if f"Point{i}" in parts:
                            point_L_filename = filename
                            information.append(point_L_filename)

                if "B" == parts[-1] and "L" in parts:
                    for i in range(self.tracknumber+1):
                        if f"Point{i}" in parts:
                            point_B_filename = filename
                            information.append(point_B_filename)

        for filename in filenames:
            parts = filename.replace(".npy", "").split("_")
            if "L" == parts[-1] and "crop" in parts:
                coorL = [int(parts[0]), int(parts[1])]
                information.append(coorL)
                break
        for filename in filenames:
            parts = filename.replace(".npy", "").split("_")
            if "B" == parts[-1] and "crop" in parts:
                coorB = [int(parts[0]), int(parts[1])]
                information.append(coorB)
                break
        # information = [filename_prefix, result_root_path,
        #                point1_L_filename, point2_L_filename, point1_B_filename, point2_B_filename, coorL, coorB]
        print(information)
        self.calculate_3d_datas_and_feature(information)

    # information = [filename_prefix,
        # def find_dis(self):
        #     obj_points = []
        #     im_points = []

        #     points3D = np.zeros((1, row * column, 3), np.float32)
        #     points3D[0, :, :2] = np.mgrid[0:row, 0:column].T.reshape(-1, 2)
        #     # Loop
        #     image_folder = glob.glob('{}'.format(self.folder_path+"/*bmp"))
        #     for files in image_folder:
        #         image = cv2.imread(files)
        #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #         ret, corners = cv2.findChessboardCorners(
        #             gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        #         if ret == True:
        #             obj_points.append(points3D)
        #             corners2 = cv2.cornerSubPix(
        #                 gray, corners, (11, 11), (-1, -1), criteria)
        #             im_points.append(corners2)

        #     ret, cam_mtx, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        #         obj_points, im_points, gray.shape[::-1], None, None)
        #     print("Distortion:\n", distCoeffs)