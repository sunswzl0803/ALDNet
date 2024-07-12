#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
import time
import os
import cv2
import numpy as np
from PIL import Image

from pbgnet import PBGnet_ONNX, PBGnet

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用一图片。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在pbgnet.py_346行左右处的PBGnet_ONNX
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   count               指定了是否进行目标的像素点计数（即面积）与比例计算
    #   name_classes        区分的种类，和json_to_dataset里面的一样，用于打印种类和数量
    #   count、name_classes仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    count           = True
    name_classes    = ["_background_", "leaf"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/alternaria.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/best"
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    if mode != "predict_onnx":
        pbgnet = PBGnet()
    else:
        yolo = PBGnet_ONNX()
    if mode == "predict":
        '''
        predict.py有几个注意点
        1、该代码无法直接进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
        具体流程可以参考get_miou_prediction.py，在get_miou_prediction.py即实现了遍历。
        2、如果想要保存，利用r_image.save("img.jpg")即可保存。
        3、如果想要原图和分割图不混合，可以把blend参数设置成False。
        4、如果想根据mask获取对应的区域，可以参考detect_image函数中，利用预测结果绘图的部分，判断每一个像素点的种类，然后根据种类获取对应的部分。
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            save_path = "img_out"
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = pbgnet.detect_image(image, count=count, name_classes=name_classes)
                r_image.save("img.png")  # rgb
                # Open the original image
                original_img = Image.open('img.png')
                # Convert the image to a palette image
                palette_img = original_img.convert('P', palette=Image.ADAPTIVE, colors=2)  # 2 colors in the palette
                # Get the histogram to identify most frequent colors (background and leaves)
                histogram = palette_img.histogram()
                colors = sorted(enumerate(histogram), key=lambda x: -x[1])  # Sort colors by frequency
                # Determine the background and leaves colors
                background_color_index = colors[0][0]
                leaves_color_index = colors[1][0]
                # Swap the colors in the palette to ensure background (black) becomes index 0 and leaves (red) becomes index 1
                palette = palette_img.getpalette()
                palette[background_color_index * 3:background_color_index * 3 + 3] = [0, 0, 0]  # Set background as black
                palette[leaves_color_index * 3:leaves_color_index * 3 + 3] = [128, 0, 0]  # Set leaves as red
                # Swap color indices to ensure background (black) becomes index 0 and leaves (red) becomes index 1
                data = palette_img.getdata()
                new_data = [1 if p == leaves_color_index else 0 for p in data]
                palette_img.putdata(new_data)
                # 下面代码其实是在deals下面的huhuan_index0he1yanse.py
                # Swap the colors of index 0 and index 1 in the palette
                index_0_color = palette[0:3]
                index_1_color = palette[3:6]
                # Swap the colors
                palette[0:3] = index_1_color
                palette[3:6] = index_0_color
                # Update the palette in the image
                palette_img.putpalette(palette)
                # Save the corrected palette image
                palette_img.save('predict_img.png')  # 这里的标签图像是反着的颜色，且index0（128，0，0）
                # r_image.save(os.path.join(save_path, os.path.basename(img)))
                # r_image.save(os.path.join(save_path, os.path.splitext(os.path.basename(img))[0] + ".png"))

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(pbgnet.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = pbgnet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm


        def convert_to_palette(input_image_path, output_image_path):
            image = Image.open(input_image_path)

            # Perform prediction using PBGNet here
            r_image = pbgnet.detect_image(image)

            # Convert the predicted image to a palette image
            palette_img = r_image.convert('P', palette=Image.ADAPTIVE, colors=2)  # 2 colors in the palette

            # Get the histogram to identify most frequent colors (background and leaves)
            histogram = palette_img.histogram()
            colors = sorted(enumerate(histogram), key=lambda x: -x[1])  # Sort colors by frequency

            # Determine the background and leaves colors
            background_color_index = colors[0][0]
            leaves_color_index = colors[1][0]
            a2_color_index = colors[2][0]
            a3_color_index = colors[3][0]
            a4_color_index = colors[4][0]
            a5_color_index = colors[5][0]
            a6_color_index = colors[6][0]
            a7_color_index = colors[7][0]
            a8_color_index = colors[8][0]
            a9_color_index = colors[9][0]
            a10_color_index = colors[10][0]
            a11_color_index = colors[11][0]
            a12_color_index = colors[12][0]
            a13_color_index = colors[13][0]
            a14_color_index = colors[14][0]
            a15_color_index = colors[15][0]
            a16_color_index = colors[16][0]
            a17_color_index = colors[17][0]
            a18_color_index = colors[18][0]
            a19_color_index = colors[19][0]
            a20_color_index = colors[20][0]
            a21_color_index = colors[21][0]
            a22_color_index = colors[22][0]
            a23_color_index = colors[23][0]
            a24_color_index = colors[24][0]
            a25_color_index = colors[25][0]
            a26_color_index = colors[26][0]
            a27_color_index = colors[27][0]
            a28_color_index = colors[28][0]
            a29_color_index = colors[29][0]
            a30_color_index = colors[30][0]
            a31_color_index = colors[31][0]
            a32_color_index = colors[32][0]
            a33_color_index = colors[33][0]
            a34_color_index = colors[34][0]
            a35_color_index = colors[35][0]
            a36_color_index = colors[36][0]
            a37_color_index = colors[37][0]
            a38_color_index = colors[38][0]
            a39_color_index = colors[39][0]
            a40_color_index = colors[40][0]
            a41_color_index = colors[41][0]
            a42_color_index = colors[42][0]
            a43_color_index = colors[43][0]
            a44_color_index = colors[44][0]
            a45_color_index = colors[45][0]
            a46_color_index = colors[46][0]
            a47_color_index = colors[47][0]
            a48_color_index = colors[48][0]
            a49_color_index = colors[49][0]
            a50_color_index = colors[50][0]
            a51_color_index = colors[51][0]
            a52_color_index = colors[52][0]
            a53_color_index = colors[53][0]
            a54_color_index = colors[54][0]
            a55_color_index = colors[55][0]
            a56_color_index = colors[56][0]
            a57_color_index = colors[57][0]
            a58_color_index = colors[58][0]
            a59_color_index = colors[59][0]
            a60_color_index = colors[60][0]
            a61_color_index = colors[61][0]
            a62_color_index = colors[62][0]
            a63_color_index = colors[63][0]
            a64_color_index = colors[64][0]
            a65_color_index = colors[65][0]
            a66_color_index = colors[66][0]
            a67_color_index = colors[67][0]
            a68_color_index = colors[68][0]
            a69_color_index = colors[69][0]
            a70_color_index = colors[70][0]
            a71_color_index = colors[71][0]
            a72_color_index = colors[72][0]
            a73_color_index = colors[73][0]
            a74_color_index = colors[74][0]
            a75_color_index = colors[75][0]
            a76_color_index = colors[76][0]
            a77_color_index = colors[77][0]
            a78_color_index = colors[78][0]
            a79_color_index = colors[79][0]
            a80_color_index = colors[80][0]
            a81_color_index = colors[81][0]
            a82_color_index = colors[82][0]
            a83_color_index = colors[83][0]
            a84_color_index = colors[84][0]
            a85_color_index = colors[85][0]
            a86_color_index = colors[86][0]
            a87_color_index = colors[87][0]
            a88_color_index = colors[88][0]
            a89_color_index = colors[89][0]
            a90_color_index = colors[90][0]
            a91_color_index = colors[91][0]
            a92_color_index = colors[92][0]
            a93_color_index = colors[93][0]
            a94_color_index = colors[94][0]
            a95_color_index = colors[95][0]
            a96_color_index = colors[96][0]
            a97_color_index = colors[97][0]
            a98_color_index = colors[98][0]
            a99_color_index = colors[99][0]
            a100_color_index = colors[100][0]
            a101_color_index = colors[101][0]
            a102_color_index = colors[102][0]
            a103_color_index = colors[103][0]
            a104_color_index = colors[104][0]
            a105_color_index = colors[105][0]
            a106_color_index = colors[106][0]
            a107_color_index = colors[107][0]
            a108_color_index = colors[108][0]
            a109_color_index = colors[109][0]
            a110_color_index = colors[110][0]
            a111_color_index = colors[111][0]
            a112_color_index = colors[112][0]
            a113_color_index = colors[113][0]
            a114_color_index = colors[114][0]
            a115_color_index = colors[115][0]
            a116_color_index = colors[116][0]
            a117_color_index = colors[117][0]
            a118_color_index = colors[118][0]
            a119_color_index = colors[119][0]
            a120_color_index = colors[120][0]
            a121_color_index = colors[121][0]
            a122_color_index = colors[122][0]
            a123_color_index = colors[123][0]
            a124_color_index = colors[124][0]
            a125_color_index = colors[125][0]
            a126_color_index = colors[126][0]
            a127_color_index = colors[127][0]
            a128_color_index = colors[128][0]
            a129_color_index = colors[129][0]
            a130_color_index = colors[130][0]
            a131_color_index = colors[131][0]
            a132_color_index = colors[132][0]
            a133_color_index = colors[133][0]
            a134_color_index = colors[134][0]
            a135_color_index = colors[135][0]
            a136_color_index = colors[136][0]
            a137_color_index = colors[137][0]
            a138_color_index = colors[138][0]
            a139_color_index = colors[139][0]
            a140_color_index = colors[140][0]
            a141_color_index = colors[141][0]
            a142_color_index = colors[142][0]
            a143_color_index = colors[143][0]
            a144_color_index = colors[144][0]
            a145_color_index = colors[145][0]
            a146_color_index = colors[146][0]
            a147_color_index = colors[147][0]
            a148_color_index = colors[148][0]
            a149_color_index = colors[149][0]
            a150_color_index = colors[150][0]
            a151_color_index = colors[151][0]
            a152_color_index = colors[152][0]
            a153_color_index = colors[153][0]
            a154_color_index = colors[154][0]
            a155_color_index = colors[155][0]
            a156_color_index = colors[156][0]
            a157_color_index = colors[157][0]
            a158_color_index = colors[158][0]
            a159_color_index = colors[159][0]
            a160_color_index = colors[160][0]
            a161_color_index = colors[161][0]
            a162_color_index = colors[162][0]
            a163_color_index = colors[163][0]
            a164_color_index = colors[164][0]
            a165_color_index = colors[165][0]
            a166_color_index = colors[166][0]
            a167_color_index = colors[167][0]
            a168_color_index = colors[168][0]
            a169_color_index = colors[169][0]
            a170_color_index = colors[170][0]
            a171_color_index = colors[171][0]
            a172_color_index = colors[172][0]
            a173_color_index = colors[173][0]
            a174_color_index = colors[174][0]
            a175_color_index = colors[175][0]
            a176_color_index = colors[176][0]
            a177_color_index = colors[177][0]
            a178_color_index = colors[178][0]
            a179_color_index = colors[179][0]
            a180_color_index = colors[180][0]
            a181_color_index = colors[181][0]
            a182_color_index = colors[182][0]
            a183_color_index = colors[183][0]
            a184_color_index = colors[184][0]
            a185_color_index = colors[185][0]
            a186_color_index = colors[186][0]
            a187_color_index = colors[187][0]
            a188_color_index = colors[188][0]
            a189_color_index = colors[189][0]
            a190_color_index = colors[190][0]
            a191_color_index = colors[191][0]
            a192_color_index = colors[192][0]
            a193_color_index = colors[193][0]
            a194_color_index = colors[194][0]
            a195_color_index = colors[195][0]
            a196_color_index = colors[196][0]
            a197_color_index = colors[197][0]
            a198_color_index = colors[198][0]
            a199_color_index = colors[199][0]
            a200_color_index = colors[200][0]
            a201_color_index = colors[201][0]
            a202_color_index = colors[202][0]
            a203_color_index = colors[203][0]
            a204_color_index = colors[204][0]
            a205_color_index = colors[205][0]
            a206_color_index = colors[206][0]
            a207_color_index = colors[207][0]
            a208_color_index = colors[208][0]
            a209_color_index = colors[209][0]
            a210_color_index = colors[210][0]
            a211_color_index = colors[211][0]
            a212_color_index = colors[212][0]
            a213_color_index = colors[213][0]
            a214_color_index = colors[214][0]
            a215_color_index = colors[215][0]
            a216_color_index = colors[216][0]
            a217_color_index = colors[217][0]
            a218_color_index = colors[218][0]
            a219_color_index = colors[219][0]
            a220_color_index = colors[220][0]
            a221_color_index = colors[221][0]
            a222_color_index = colors[222][0]
            a223_color_index = colors[223][0]
            a224_color_index = colors[224][0]
            a225_color_index = colors[225][0]
            a226_color_index = colors[226][0]
            a227_color_index = colors[227][0]
            a228_color_index = colors[228][0]
            a229_color_index = colors[229][0]
            a230_color_index = colors[230][0]
            a231_color_index = colors[231][0]
            a232_color_index = colors[232][0]
            a233_color_index = colors[233][0]
            a234_color_index = colors[234][0]
            a235_color_index = colors[235][0]
            a236_color_index = colors[236][0]
            a237_color_index = colors[237][0]
            a238_color_index = colors[238][0]
            a239_color_index = colors[239][0]
            a240_color_index = colors[240][0]
            a241_color_index = colors[241][0]
            a242_color_index = colors[242][0]
            a243_color_index = colors[243][0]
            a244_color_index = colors[244][0]
            a245_color_index = colors[245][0]
            a246_color_index = colors[246][0]
            a247_color_index = colors[247][0]
            a248_color_index = colors[248][0]
            a249_color_index = colors[249][0]
            a250_color_index = colors[250][0]
            a251_color_index = colors[251][0]
            a252_color_index = colors[252][0]
            a253_color_index = colors[253][0]
            a254_color_index = colors[254][0]
            a255_color_index = colors[255][0]

            # Swap the colors in the palette to ensure background (black) becomes index 0 and leaves (red) becomes index 1
            palette = palette_img.getpalette()
            palette[background_color_index * 3:background_color_index * 3 + 3] = [0, 0, 0]  # Set background as black
            palette[leaves_color_index * 3:leaves_color_index * 3 + 3] = [128, 0, 0]  # Set leaves as red
            palette[a2_color_index * 3:a2_color_index * 3 + 3] = [0, 128, 0]  # Set leaves as red
            palette[a3_color_index * 3:a3_color_index * 3 + 3] = [128, 128, 0]  # Set leaves as red
            palette[a4_color_index * 3:a4_color_index * 3 + 3] = [0, 0, 128]  # Set leaves as red
            palette[a5_color_index * 3:a5_color_index * 3 + 3] = [128, 0, 128]  # Set leaves as red
            palette[a6_color_index * 3:a6_color_index * 3 + 3] = [0, 128, 128]  # Set leaves as red
            palette[a7_color_index * 3:a7_color_index * 3 + 3] = [128, 128, 128]  # Set leaves as red
            palette[a8_color_index * 3:a8_color_index * 3 + 3] = [64, 0, 0]  # Set leaves as red
            palette[a9_color_index * 3:a9_color_index * 3 + 3] = [192, 0, 0]  # Set leaves as red
            palette[a10_color_index * 3:a10_color_index * 3 + 3] = [64, 128, 0]  # Set leaves as red

            palette[a11_color_index * 3:a11_color_index * 3 + 3] = [192, 128, 0]  # Set leaves as red
            palette[a12_color_index * 3:a12_color_index * 3 + 3] = [64, 0, 128]  # Set leaves as red
            palette[a13_color_index * 3:a13_color_index * 3 + 3] = [192, 0, 128]  # Set leaves as red
            palette[a14_color_index * 3:a14_color_index * 3 + 3] = [64, 128, 128]  # Set leaves as red
            palette[a15_color_index * 3:a15_color_index * 3 + 3] = [192, 128, 128]  # Set leaves as red
            palette[a16_color_index * 3:a16_color_index * 3 + 3] = [0, 64, 0]  # Set leaves as red
            palette[a17_color_index * 3:a17_color_index * 3 + 3] = [128, 64, 0]  # Set leaves as red
            palette[a18_color_index * 3:a18_color_index * 3 + 3] = [0, 192, 0]  # Set leaves as red
            palette[a19_color_index * 3:a19_color_index * 3 + 3] = [128, 192, 0]  # Set leaves as red
            palette[a20_color_index * 3:a20_color_index * 3 + 3] = [0, 64, 128]  # Set leaves as red

            palette[a21_color_index * 3:a21_color_index * 3 + 3] = [128, 64, 128]  # Set leaves as red
            palette[a22_color_index * 3:a22_color_index * 3 + 3] = [0, 192, 128]  # Set leaves as red
            palette[a23_color_index * 3:a23_color_index * 3 + 3] = [128, 192, 128]  # Set leaves as red
            palette[a24_color_index * 3:a24_color_index * 3 + 3] = [64, 64, 0]  # Set leaves as red
            palette[a25_color_index * 3:a25_color_index * 3 + 3] = [192, 64, 0]  # Set leaves as red
            palette[a26_color_index * 3:a26_color_index * 3 + 3] = [64, 192, 0]  # Set leaves as red
            palette[a27_color_index * 3:a27_color_index * 3 + 3] = [192, 192, 0]  # Set leaves as red
            palette[a28_color_index * 3:a28_color_index * 3 + 3] = [64, 64, 128]  # Set leaves as red
            palette[a29_color_index * 3:a29_color_index * 3 + 3] = [192, 64, 128]  # Set leaves as red
            palette[a30_color_index * 3:a30_color_index * 3 + 3] = [64, 192, 128]  # Set leaves as red

            palette[a31_color_index * 3:a31_color_index * 3 + 3] = [192, 192, 128]  # Set leaves as red
            palette[a32_color_index * 3:a32_color_index * 3 + 3] = [0, 0, 64]  # Set leaves as red
            palette[a33_color_index * 3:a33_color_index * 3 + 3] = [128, 0, 64]  # Set leaves as red
            palette[a34_color_index * 3:a34_color_index * 3 + 3] = [0, 128, 64]  # Set leaves as red
            palette[a35_color_index * 3:a35_color_index * 3 + 3] = [128, 128, 64]  # Set leaves as red
            palette[a36_color_index * 3:a36_color_index * 3 + 3] = [0, 0, 192]  # Set leaves as red
            palette[a37_color_index * 3:a37_color_index * 3 + 3] = [128, 0, 192]  # Set leaves as red
            palette[a38_color_index * 3:a38_color_index * 3 + 3] = [0, 128, 192]  # Set leaves as red
            palette[a39_color_index * 3:a39_color_index * 3 + 3] = [128, 128, 192]  # Set leaves as red
            palette[a40_color_index * 3:a40_color_index * 3 + 3] = [64, 0, 64]  # Set leaves as red

            palette[a41_color_index * 3:a41_color_index * 3 + 3] = [192, 0, 64]  # Set leaves as red
            palette[a42_color_index * 3:a42_color_index * 3 + 3] = [64, 128, 64]  # Set leaves as red
            palette[a43_color_index * 3:a43_color_index * 3 + 3] = [192, 128, 64]  # Set leaves as red
            palette[a44_color_index * 3:a44_color_index * 3 + 3] = [64, 0, 192]  # Set leaves as red
            palette[a45_color_index * 3:a45_color_index * 3 + 3] = [192, 0, 192]  # Set leaves as red
            palette[a46_color_index * 3:a46_color_index * 3 + 3] = [64, 128, 192]  # Set leaves as red
            palette[a47_color_index * 3:a47_color_index * 3 + 3] = [192, 128, 192]  # Set leaves as red
            palette[a48_color_index * 3:a48_color_index * 3 + 3] = [0, 64, 64]  # Set leaves as red
            palette[a49_color_index * 3:a49_color_index * 3 + 3] = [128, 64, 64]  # Set leaves as re
            palette[a50_color_index * 3:a50_color_index * 3 + 3] = [0, 192, 64]  # Set leaves as red

            palette[a51_color_index * 3:a51_color_index * 3 + 3] = [128, 192, 64]  # Set leaves as red
            palette[a52_color_index * 3:a52_color_index * 3 + 3] = [0, 64, 192]  # Set leaves as red
            palette[a53_color_index * 3:a53_color_index * 3 + 3] = [128, 64, 192]  # Set leaves as red
            palette[a54_color_index * 3:a54_color_index * 3 + 3] = [0, 192, 192]  # Set leaves as red
            palette[a55_color_index * 3:a55_color_index * 3 + 3] = [128, 192, 192]  # Set leaves as red
            palette[a56_color_index * 3:a56_color_index * 3 + 3] = [64, 64, 64]  # Set leaves as red
            palette[a57_color_index * 3:a57_color_index * 3 + 3] = [192, 64, 64]  # Set leaves as red
            palette[a58_color_index * 3:a58_color_index * 3 + 3] = [64, 192, 64]  # Set leaves as red
            palette[a59_color_index * 3:a59_color_index * 3 + 3] = [192, 192, 64]  # Set leaves as red
            palette[a60_color_index * 3:a60_color_index * 3 + 3] = [64, 64, 192]  # Set leaves as red

            palette[a61_color_index * 3:a61_color_index * 3 + 3] = [192, 64, 192]  # Set leaves as red
            palette[a62_color_index * 3:a62_color_index * 3 + 3] = [64, 192, 192]  # Set leaves as red
            palette[a63_color_index * 3:a63_color_index * 3 + 3] = [192, 192, 192]  # Set leaves as red
            palette[a64_color_index * 3:a64_color_index * 3 + 3] = [32, 0, 0]  # Set leaves as red
            palette[a65_color_index * 3:a65_color_index * 3 + 3] = [160, 0, 0]  # Set leaves as red
            palette[a66_color_index * 3:a66_color_index * 3 + 3] = [32, 128, 0]  # Set leaves as red
            palette[a67_color_index * 3:a67_color_index * 3 + 3] = [160, 128, 0]  # Set leaves as red
            palette[a68_color_index * 3:a68_color_index * 3 + 3] = [32, 0, 128]  # Set leaves as red
            palette[a69_color_index * 3:a69_color_index * 3 + 3] = [160, 0, 128]  # Set leaves as red
            palette[a70_color_index * 3:a70_color_index * 3 + 3] = [32, 128, 128]  # Set leaves as red

            palette[a71_color_index * 3:a71_color_index * 3 + 3] = [160, 128, 128]  # Set leaves as red
            palette[a72_color_index * 3:a72_color_index * 3 + 3] = [96, 0, 0]  # Set leaves as red
            palette[a73_color_index * 3:a73_color_index * 3 + 3] = [224, 0, 0]  # Set leaves as red
            palette[a74_color_index * 3:a74_color_index * 3 + 3] = [96, 128, 0]  # Set leaves as red
            palette[a75_color_index * 3:a75_color_index * 3 + 3] = [224, 128, 0]  # Set leaves as red
            palette[a76_color_index * 3:a76_color_index * 3 + 3] = [96, 0, 128]  # Set leaves as red
            palette[a77_color_index * 3:a77_color_index * 3 + 3] = [224, 0, 128]  # Set leaves as red
            palette[a78_color_index * 3:a78_color_index * 3 + 3] = [96, 128, 128]  # Set leaves as red
            palette[a79_color_index * 3:a79_color_index * 3 + 3] = [224, 128, 128]  # Set leaves as red
            palette[a80_color_index * 3:a80_color_index * 3 + 3] = [32, 64, 0]  # Set leaves as red

            palette[a81_color_index * 3:a81_color_index * 3 + 3] = [160, 64, 0]  # Set leaves as red
            palette[a82_color_index * 3:a82_color_index * 3 + 3] = [32, 192, 0]  # Set leaves as red
            palette[a83_color_index * 3:a83_color_index * 3 + 3] = [160, 192, 0]  # Set leaves as red
            palette[a84_color_index * 3:a84_color_index * 3 + 3] = [32, 64, 128]  # Set leaves as red
            palette[a85_color_index * 3:a85_color_index * 3 + 3] = [160, 64, 128]  # Set leaves as red
            palette[a86_color_index * 3:a86_color_index * 3 + 3] = [32, 192, 128]  # Set leaves as red
            palette[a87_color_index * 3:a87_color_index * 3 + 3] = [160, 192, 128]  # Set leaves as red
            palette[a88_color_index * 3:a88_color_index * 3 + 3] = [96, 64, 0]  # Set leaves as red
            palette[a89_color_index * 3:a89_color_index * 3 + 3] = [224, 64, 0]  # Set leaves as red
            palette[a90_color_index * 3:a90_color_index * 3 + 3] = [96, 192, 0]  # Set leaves as red

            palette[a91_color_index * 3:a91_color_index * 3 + 3] = [224, 192, 0]  # Set leaves as red
            palette[a92_color_index * 3:a92_color_index * 3 + 3] = [96, 64, 128]  # Set leaves as red
            palette[a93_color_index * 3:a93_color_index * 3 + 3] = [224, 64, 128]  # Set leaves as red
            palette[a94_color_index * 3:a94_color_index * 3 + 3] = [96, 192, 128]  # Set leaves as red
            palette[a95_color_index * 3:a95_color_index * 3 + 3] = [224, 192, 128]  # Set leaves as red
            palette[a96_color_index * 3:a96_color_index * 3 + 3] = [32, 0, 64]  # Set leaves as red
            palette[a97_color_index * 3:a97_color_index * 3 + 3] = [160, 0, 64]  # Set leaves as red
            palette[a98_color_index * 3:a98_color_index * 3 + 3] = [32, 128, 64]  # Set leaves as red
            palette[a99_color_index * 3:a99_color_index * 3 + 3] = [160, 128, 64]  # Set leaves as red
            palette[a100_color_index * 3:a100_color_index * 3 + 3] = [32, 0, 192]  # Set leaves as red

            palette[a101_color_index * 3:a101_color_index * 3 + 3] = [160, 0, 192]  # Set leaves as red
            palette[a102_color_index * 3:a102_color_index * 3 + 3] = [32, 128, 192]  # Set leaves as red
            palette[a103_color_index * 3:a103_color_index * 3 + 3] = [160, 128, 192]  # Set leaves as red
            palette[a104_color_index * 3:a104_color_index * 3 + 3] = [96, 0, 64]  # Set leaves as red
            palette[a105_color_index * 3:a105_color_index * 3 + 3] = [224, 0, 64]  # Set leaves as red
            palette[a106_color_index * 3:a106_color_index * 3 + 3] = [96, 128, 64]  # Set leaves as red
            palette[a107_color_index * 3:a107_color_index * 3 + 3] = [224, 128, 64]  # Set leaves as red
            palette[a108_color_index * 3:a108_color_index * 3 + 3] = [96, 0, 192]  # Set leaves as red
            palette[a109_color_index * 3:a109_color_index * 3 + 3] = [224, 0, 192]  # Set leaves as red
            palette[a110_color_index * 3:a110_color_index * 3 + 3] = [96, 128, 192]  # Set leaves as red

            palette[a111_color_index * 3:a111_color_index * 3 + 3] = [224, 128, 192]  # Set leaves as red
            palette[a112_color_index * 3:a112_color_index * 3 + 3] = [32, 64, 64]  # Set leaves as red
            palette[a113_color_index * 3:a113_color_index * 3 + 3] = [160, 64, 64]  # Set leaves as red
            palette[a114_color_index * 3:a114_color_index * 3 + 3] = [32, 192, 64]  # Set leaves as red
            palette[a115_color_index * 3:a115_color_index * 3 + 3] = [160, 192, 64]  # Set leaves as red
            palette[a116_color_index * 3:a116_color_index * 3 + 3] = [32, 64, 192]  # Set leaves as red
            palette[a117_color_index * 3:a117_color_index * 3 + 3] = [160, 64, 192]  # Set leaves as red
            palette[a118_color_index * 3:a118_color_index * 3 + 3] = [32, 192, 192]  # Set leaves as red
            palette[a119_color_index * 3:a119_color_index * 3 + 3] = [160, 192, 192]  # Set leaves as red
            palette[a120_color_index * 3:a120_color_index * 3 + 3] = [96, 64, 64]  # Set leaves as red

            palette[a121_color_index * 3:a121_color_index * 3 + 3] = [224, 64, 64]  # Set leaves as red
            palette[a122_color_index * 3:a122_color_index * 3 + 3] = [96, 192, 64]  # Set leaves as red
            palette[a123_color_index * 3:a123_color_index * 3 + 3] = [224, 192, 64]  # Set leaves as red
            palette[a124_color_index * 3:a124_color_index * 3 + 3] = [96, 64, 192]  # Set leaves as red
            palette[a125_color_index * 3:a125_color_index * 3 + 3] = [224, 64, 192]  # Set leaves as red
            palette[a126_color_index * 3:a126_color_index * 3 + 3] = [96, 192, 192]  # Set leaves as red
            palette[a127_color_index * 3:a127_color_index * 3 + 3] = [224, 192, 192]  # Set leaves as red
            palette[a128_color_index * 3:a128_color_index * 3 + 3] = [0, 32, 0]  # Set leaves as red
            palette[a129_color_index * 3:a129_color_index * 3 + 3] = [128, 32, 0]  # Set leaves as red
            palette[a130_color_index * 3:a130_color_index * 3 + 3] = [0, 160, 0]  # Set leaves as red

            palette[a131_color_index * 3:a131_color_index * 3 + 3] = [128, 160, 0]  # Set leaves as red
            palette[a132_color_index * 3:a132_color_index * 3 + 3] = [0, 32, 128]  # Set leaves as red
            palette[a133_color_index * 3:a133_color_index * 3 + 3] = [128, 32, 128]  # Set leaves as red
            palette[a134_color_index * 3:a134_color_index * 3 + 3] = [0, 160, 128]  # Set leaves as red
            palette[a135_color_index * 3:a135_color_index * 3 + 3] = [128, 160, 128]  # Set leaves as red
            palette[a136_color_index * 3:a136_color_index * 3 + 3] = [64, 32, 0]  # Set leaves as red
            palette[a137_color_index * 3:a137_color_index * 3 + 3] = [192, 32, 0]  # Set leaves as red
            palette[a138_color_index * 3:a138_color_index * 3 + 3] = [64, 160, 0]  # Set leaves as red
            palette[a139_color_index * 3:a139_color_index * 3 + 3] = [192, 160, 0]  # Set leaves as red
            palette[a140_color_index * 3:a140_color_index * 3 + 3] = [64, 32, 128]  # Set leaves as red

            palette[a141_color_index * 3:a141_color_index * 3 + 3] = [192, 32, 128]  # Set leaves as red
            palette[a142_color_index * 3:a142_color_index * 3 + 3] = [64, 160, 128]  # Set leaves as red
            palette[a143_color_index * 3:a143_color_index * 3 + 3] = [192, 160, 128]  # Set leaves as red
            palette[a144_color_index * 3:a144_color_index * 3 + 3] = [0, 96, 0]  # Set leaves as red
            palette[a145_color_index * 3:a145_color_index * 3 + 3] = [128, 96, 0]  # Set leaves as red
            palette[a146_color_index * 3:a146_color_index * 3 + 3] = [0, 224, 0]  # Set leaves as red
            palette[a147_color_index * 3:a147_color_index * 3 + 3] = [128, 224, 0]  # Set leaves as red
            palette[a148_color_index * 3:a148_color_index * 3 + 3] = [0, 96, 128]  # Set leaves as red
            palette[a149_color_index * 3:a149_color_index * 3 + 3] = [128, 96, 128]  # Set leaves as red
            palette[a150_color_index * 3:a150_color_index * 3 + 3] = [0, 224, 128]  # Set leaves as red

            palette[a151_color_index * 3:a151_color_index * 3 + 3] = [128, 224, 128]  # Set leaves as red
            palette[a152_color_index * 3:a152_color_index * 3 + 3] = [64, 96, 0]  # Set leaves as red
            palette[a153_color_index * 3:a153_color_index * 3 + 3] = [192, 96, 0]  # Set leaves as red
            palette[a154_color_index * 3:a154_color_index * 3 + 3] = [64, 224, 0]  # Set leaves as red
            palette[a155_color_index * 3:a155_color_index * 3 + 3] = [192, 224, 0]  # Set leaves as red
            palette[a156_color_index * 3:a156_color_index * 3 + 3] = [64, 96, 128]  # Set leaves as red
            palette[a157_color_index * 3:a157_color_index * 3 + 3] = [192, 96, 128]  # Set leaves as red
            palette[a158_color_index * 3:a158_color_index * 3 + 3] = [64, 224, 128]  # Set leaves as red
            palette[a159_color_index * 3:a159_color_index * 3 + 3] = [192, 224, 128]  # Set leaves as red
            palette[a160_color_index * 3:a160_color_index * 3 + 3] = [0, 32, 64]  # Set leaves as red

            palette[a161_color_index * 3:a161_color_index * 3 + 3] = [128, 32, 64]  # Set leaves as red
            palette[a162_color_index * 3:a162_color_index * 3 + 3] = [0, 160, 64]  # Set leaves as red
            palette[a163_color_index * 3:a163_color_index * 3 + 3] = [128, 160, 64]  # Set leaves as red
            palette[a164_color_index * 3:a164_color_index * 3 + 3] = [0, 32, 192]  # Set leaves as red
            palette[a165_color_index * 3:a165_color_index * 3 + 3] = [128, 32, 192]  # Set leaves as red
            palette[a166_color_index * 3:a166_color_index * 3 + 3] = [0, 160, 192]  # Set leaves as red
            palette[a167_color_index * 3:a167_color_index * 3 + 3] = [128, 160, 192]  # Set leaves as red
            palette[a168_color_index * 3:a168_color_index * 3 + 3] = [64, 32, 64]  # Set leaves as red
            palette[a169_color_index * 3:a169_color_index * 3 + 3] = [192, 32, 64]  # Set leaves as red
            palette[a170_color_index * 3:a170_color_index * 3 + 3] = [64, 160, 64]  # Set leaves as red

            palette[a171_color_index * 3:a171_color_index * 3 + 3] = [192, 160, 64]  # Set leaves as red
            palette[a172_color_index * 3:a172_color_index * 3 + 3] = [64, 32, 192]  # Set leaves as red
            palette[a173_color_index * 3:a173_color_index * 3 + 3] = [192, 32, 192]  # Set leaves as red
            palette[a174_color_index * 3:a174_color_index * 3 + 3] = [64, 160, 192]  # Set leaves as red
            palette[a175_color_index * 3:a175_color_index * 3 + 3] = [192, 160, 192]  # Set leaves as red
            palette[a176_color_index * 3:a176_color_index * 3 + 3] = [0, 96, 64]  # Set leaves as red
            palette[a177_color_index * 3:a177_color_index * 3 + 3] = [128, 96, 64]  # Set leaves as red
            palette[a178_color_index * 3:a178_color_index * 3 + 3] = [0, 224, 64]  # Set leaves as red
            palette[a179_color_index * 3:a179_color_index * 3 + 3] = [128, 224, 64]  # Set leaves as red
            palette[a180_color_index * 3:a180_color_index * 3 + 3] = [0, 96, 192]  # Set leaves as red

            palette[a181_color_index * 3:a181_color_index * 3 + 3] = [128, 96, 192]  # Set leaves as red
            palette[a182_color_index * 3:a182_color_index * 3 + 3] = [0, 224, 192]  # Set leaves as red
            palette[a183_color_index * 3:a183_color_index * 3 + 3] = [128, 224, 192]  # Set leaves as red
            palette[a184_color_index * 3:a184_color_index * 3 + 3] = [64, 96, 64]  # Set leaves as red
            palette[a185_color_index * 3:a185_color_index * 3 + 3] = [192, 96, 64]  # Set leaves as red
            palette[a186_color_index * 3:a186_color_index * 3 + 3] = [64, 224, 64]  # Set leaves as red
            palette[a187_color_index * 3:a187_color_index * 3 + 3] = [192, 224, 64]  # Set leaves as red
            palette[a188_color_index * 3:a188_color_index * 3 + 3] = [64, 96, 192]  # Set leaves as red
            palette[a189_color_index * 3:a189_color_index * 3 + 3] = [192, 96, 192]  # Set leaves as red
            palette[a190_color_index * 3:a190_color_index * 3 + 3] = [64, 224, 192]  # Set leaves as red

            palette[a191_color_index * 3:a191_color_index * 3 + 3] = [192, 224, 192]  # Set leaves as red
            palette[a192_color_index * 3:a192_color_index * 3 + 3] = [32, 32, 0]  # Set leaves as red
            palette[a193_color_index * 3:a193_color_index * 3 + 3] = [160, 32, 0]  # Set leaves as red
            palette[a194_color_index * 3:a194_color_index * 3 + 3] = [32, 160, 0]  # Set leaves as red
            palette[a195_color_index * 3:a195_color_index * 3 + 3] = [160, 160, 0]  # Set leaves as red
            palette[a196_color_index * 3:a196_color_index * 3 + 3] = [32, 32, 128]  # Set leaves as red
            palette[a197_color_index * 3:a197_color_index * 3 + 3] = [160, 32, 128]  # Set leaves as red
            palette[a198_color_index * 3:a198_color_index * 3 + 3] = [32, 160, 128]  # Set leaves as red
            palette[a199_color_index * 3:a199_color_index * 3 + 3] = [160, 160, 128]  # Set leaves as red
            palette[a200_color_index * 3:a200_color_index * 3 + 3] = [96, 32, 0]  # Set leaves as red

            palette[a201_color_index * 3:a201_color_index * 3 + 3] = [224, 32, 0]  # Set leaves as red
            palette[a202_color_index * 3:a202_color_index * 3 + 3] = [96, 160, 0]  # Set leaves as red
            palette[a203_color_index * 3:a203_color_index * 3 + 3] = [224, 160, 0]  # Set leaves as red
            palette[a204_color_index * 3:a204_color_index * 3 + 3] = [96, 32, 128]  # Set leaves as red
            palette[a205_color_index * 3:a205_color_index * 3 + 3] = [224, 32, 128]  # Set leaves as red
            palette[a206_color_index * 3:a206_color_index * 3 + 3] = [96, 160, 128]  # Set leaves as red
            palette[a207_color_index * 3:a207_color_index * 3 + 3] = [224, 160, 128]  # Set leaves as red
            palette[a208_color_index * 3:a208_color_index * 3 + 3] = [32, 96, 0]  # Set leaves as red
            palette[a209_color_index * 3:a209_color_index * 3 + 3] = [160, 96, 0]  # Set leaves as red
            palette[a210_color_index * 3:a210_color_index * 3 + 3] = [32, 224, 0]  # Set leaves as red

            palette[a211_color_index * 3:a211_color_index * 3 + 3] = [160, 224, 0]  # Set leaves as red
            palette[a212_color_index * 3:a212_color_index * 3 + 3] = [32, 96, 128]  # Set leaves as red
            palette[a213_color_index * 3:a213_color_index * 3 + 3] = [160, 96, 128]  # Set leaves as red
            palette[a214_color_index * 3:a214_color_index * 3 + 3] = [32, 224, 128]  # Set leaves as red
            palette[a215_color_index * 3:a215_color_index * 3 + 3] = [160, 224, 128]  # Set leaves as red
            palette[a216_color_index * 3:a216_color_index * 3 + 3] = [96, 96, 0]  # Set leaves as red
            palette[a217_color_index * 3:a217_color_index * 3 + 3] = [224, 96, 0]  # Set leaves as red
            palette[a218_color_index * 3:a218_color_index * 3 + 3] = [96, 224, 0]  # Set leaves as red
            palette[a219_color_index * 3:a219_color_index * 3 + 3] = [224, 224, 0]  # Set leaves as red
            palette[a220_color_index * 3:a220_color_index * 3 + 3] = [96, 96, 128]  # Set leaves as red

            palette[a221_color_index * 3:a221_color_index * 3 + 3] = [224, 96, 128]  # Set leaves as red
            palette[a222_color_index * 3:a222_color_index * 3 + 3] = [96, 224, 128]  # Set leaves as red
            palette[a223_color_index * 3:a223_color_index * 3 + 3] = [224, 224, 128]  # Set leaves as red
            palette[a224_color_index * 3:a224_color_index * 3 + 3] = [32, 32, 64]  # Set leaves as red
            palette[a225_color_index * 3:a225_color_index * 3 + 3] = [160, 32, 64]  # Set leaves as red
            palette[a226_color_index * 3:a226_color_index * 3 + 3] = [32, 160, 64]  # Set leaves as red
            palette[a227_color_index * 3:a227_color_index * 3 + 3] = [160, 160, 64]  # Set leaves as red
            palette[a228_color_index * 3:a228_color_index * 3 + 3] = [32, 32, 192]  # Set leaves as red
            palette[a229_color_index * 3:a229_color_index * 3 + 3] = [160, 32, 192]  # Set leaves as red
            palette[a230_color_index * 3:a230_color_index * 3 + 3] = [32, 160, 192]  # Set leaves as red

            palette[a231_color_index * 3:a231_color_index * 3 + 3] = [160, 160, 192]  # Set leaves as re
            palette[a232_color_index * 3:a232_color_index * 3 + 3] = [96, 32, 64]  # Set leaves as red
            palette[a233_color_index * 3:a233_color_index * 3 + 3] = [224, 32, 64]  # Set leaves as red
            palette[a234_color_index * 3:a234_color_index * 3 + 3] = [96, 160, 64]  # Set leaves as red
            palette[a235_color_index * 3:a235_color_index * 3 + 3] = [224, 160, 64]  # Set leaves as red
            palette[a236_color_index * 3:a236_color_index * 3 + 3] = [96, 32, 192]  # Set leaves as red
            palette[a237_color_index * 3:a237_color_index * 3 + 3] = [224, 32, 192]  # Set leaves as red
            palette[a238_color_index * 3:a238_color_index * 3 + 3] = [96, 160, 192]  # Set leaves as red
            palette[a239_color_index * 3:a239_color_index * 3 + 3] = [224, 160, 192]  # Set leaves as red
            palette[a240_color_index * 3:a240_color_index * 3 + 3] = [32, 96, 64]  # Set leaves as red

            palette[a241_color_index * 3:a241_color_index * 3 + 3] = [160, 96, 64]  # Set leaves as red
            palette[a242_color_index * 3:a242_color_index * 3 + 3] = [32, 224, 64]  # Set leaves as red
            palette[a243_color_index * 3:a243_color_index * 3 + 3] = [160, 224, 64]  # Set leaves as red
            palette[a244_color_index * 3:a244_color_index * 3 + 3] = [32, 96, 192]  # Set leaves as red
            palette[a245_color_index * 3:a245_color_index * 3 + 3] = [160, 96, 192]  # Set leaves as red
            palette[a246_color_index * 3:a246_color_index * 3 + 3] = [32, 224, 192]  # Set leaves as red
            palette[a247_color_index * 3:a247_color_index * 3 + 3] = [160, 224, 192]  # Set leaves as red
            palette[a248_color_index * 3:a248_color_index * 3 + 3] = [96, 96, 64]  # Set leaves as red
            palette[a249_color_index * 3:a249_color_index * 3 + 3] = [224, 96, 64]  # Set leaves as red
            palette[a250_color_index * 3:a250_color_index * 3 + 3] = [96, 224, 64]  # Set leaves as red

            palette[a251_color_index * 3:a251_color_index * 3 + 3] = [224, 224, 64]  # Set leaves as red
            palette[a252_color_index * 3:a252_color_index * 3 + 3] = [96, 96, 192]  # Set leaves as red
            palette[a253_color_index * 3:a253_color_index * 3 + 3] = [224, 96, 192]  # Set leaves as red
            palette[a254_color_index * 3:a254_color_index * 3 + 3] = [96, 224, 192]  # Set leaves as red
            palette[a255_color_index * 3:a255_color_index * 3 + 3] = [255, 255, 255]  # Set leaves as red

            data = palette_img.getdata()
            new_data = [1 if p == leaves_color_index else 0 for p in data]
            palette_img.putdata(new_data)
            # ----------------------------------------------------

            # Swap the colors of index 0 and index 1 in the palette
            index_0_color = palette[0:3]
            index_1_color = palette[3:6]

            # Swap the colors
            palette[0:3] = index_1_color
            palette[3:6] = index_0_color
            # Update the palette in the image
            palette_img.putpalette(palette)

            # Save the corrected palette image
            palette_img.save(output_image_path)


        if not os.path.exists(dir_save_path):
            os.makedirs(dir_save_path)

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                output_path = os.path.join(dir_save_path, os.path.splitext(img_name)[0] + ".png")

                # Convert each image to palette format and save it
                convert_to_palette(image_path, output_path)
    #     img_names = os.listdir(dir_origin_path)
    #     for img_name in tqdm(img_names):
    #         if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
    #             image_path  = os.path.join(dir_origin_path, img_name)
    #             image       = Image.open(image_path)
    #             r_image     = pbgnet.detect_image(image)
    #             if not os.path.exists(dir_save_path):
    #                 os.makedirs(dir_save_path)
    #             r_image.save(os.path.join(dir_save_path, img_name))
    # elif mode == "export_onnx":
    #     pbgnet.convert_to_onnx(simplify, onnx_save_path)
    # elif mode == "export_onnx":
    #     pbgnet.convert_to_onnx(simplify, onnx_save_path)
                
    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
