import os

for emotion in ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']:
    os.mkdir(os.path.join('/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/VALID/frames/' + emotion))

    for filename in os.listdir('/home/ivpl-d28/Pycharmprojects/FER/AFEW/Val_AFEW/' + emotion):
        if not os.path.exists(os.path.join('/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/VALID/frames/' + emotion + "/" + str(os.path.splitext(filename)[0]))):
            os.mkdir(os.path.join('/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/VALID/frames/' + emotion + "/" + str(os.path.splitext(filename)[0])))

        command = "ffmpeg -r 1 -i /home/ivpl-d28/Pycharmprojects/FER/AFEW/Val_AFEW/" + emotion + "/" + str(filename) + " -r 1 '/home/ivpl-d28/Pycharmprojects/FER/AFEW/PREPROCESSED_AFEW/VALID/frames/" + emotion + "/" + str(os.path.splitext(filename)[0]) + "/%03d.png'"
        os.system(command=command)
