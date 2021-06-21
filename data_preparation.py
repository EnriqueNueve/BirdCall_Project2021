import os
from pydub import AudioSegment

def convert1(path):
    # Change working directory
    os.chdir(path)
    audio_files = os.listdir()

    for file in audio_files:
        # spliting the file into the name and the extension
        name, ext = os.path.splitext(file)
        if ext == ".ogg":
            ogg_sound = AudioSegment.from_ogg(file)
            # rename them using the old name + ".wav"
            ogg_sound.export("{0}.wav".format(name), format="wav")

def delete_ogg(path):
    dir_name = path
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".ogg"):
            os.remove(os.path.join(dir_name, item))


def convert_ogg(path):
    # os.chdir(path)
    # audio_files = os.listdir()
    # list = []
    if os.path.exists(path):
        file_dirs = sorted(os.listdir(path))
        for dir in file_dirs:
            if os.path.isdir(path + dir):
                files = os.listdir(path + dir)
                if not os.path.exists("data/converted_audio/" + dir):
                    os.makedirs("data/converted_audio/" + dir)
                for file in files:
                    name, ext = os.path.splitext(file)
                    if ext != ".ogg":
                        print("Skipping " + dir + "/" + file)
                        continue
                    # if ext == ".ogg":
                    print("Converting " + dir + "/" + file)
                    ogg_sound = AudioSegment.from_ogg(path + dir + "/" + file)
                    # rename them using the old name + ".wav"
                    ogg_sound.export("data/converted_audio/" + dir + "/" + name + ".wav", format="wav")
                print("Finished convert " + dir + ".")


if __name__ == '__main__':
    convert_ogg('data/train_short_audio')
     # delete_ogg('data/train_short_audio/aldfly')

