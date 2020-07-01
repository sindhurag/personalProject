'''
Download voice recordings from s3 bucket and
Convert RAW files to WAV format for further processing

'''

import subprocess
import os
import shutil


S3_BUCKET = "s3://" #<add the s3 path>

OUT_FOLDER = "Users/" #<add path to output folder>

s3_CMD = "aws s3 sync "
SOX_CMD_PART1 = "sox -t raw -r 16000 -b 16 -L -e signed  "
SOX_CMD_PART2 = " -t wav "

def removeTempStorage(dirList):
    if '.DS_Store' in dirList:
        dirList.remove('.DS_Store')

# Read voice files from the s3 bucket
def readAudioFilesFroms3(outFolder):
    try:
        s3_request = ''
        s3_request = s3_CMD + S3_BUCKET + " " + outFolder
        subprocess.call(s3_request, shell=True)
        print('s3 done')
    except:
        print('Some error occurred')
    finally:
        # Save the workbook
        print("---------------s3 reading done-------------")

#-----------------------

def rawToWav(folderPath, files):

    for f in files:
        fileType = f.split(".")
        if (fileType[1].upper()  == 'RAW'):
            rawFile = folderPath + f
            wavFile = folderPath + fileType[0] + ".wav"
            print(rawFile, wavFile)

            # Convert if it is a new file
            if not (os.path.exists(wavFile)):
                sox_cmd = SOX_CMD_PART1 + rawFile + SOX_CMD_PART2 + wavFile
                print(sox_cmd)

                subprocess.call(sox_cmd, shell=True)

    print("Files converted")
#-----------------------


# Convert RAW to WAV files
def convertRawToWav(outFolder):

    folders = os.listdir(outFolder)
    removeTempStorage(folders)

    # folders organised by date
    for folder in folders:
        folderPath = outFolder + folder + "/"
        files = os.listdir(folderPath)
        removeTempStorage(files)
        rawToWav(folderPath, files)
        print("--- RAW to WAV done for ---- " + folder )

#-----------------------

def moveNewFilesForReview(outFolder, reviewFolder):

    folders = os.listdir(outFolder)
    removeTempStorage(folders)
    # folders organised by date
    for folder in folders:
        folderPath = outFolder + folder + "/"
        files = os.listdir(folderPath)
        removeTempStorage(files)

        # make directory in review folder
        os.makedirs(reviewFolder + folder, exist_ok=True)

        for f in files:
            filepath = outFolder + folder + "/" + f
            fileInReviewFolder = reviewFolder + folder + "/" + f
            print(fileInReviewFolder)
            if not (os.path.exists(fileInReviewFolder)):
                print("File does not exist")
                shutil.copy(filepath, fileInReviewFolder)

#-----------------------

# Main method
if __name__ == '__main__':

    # Download the audio files from s3 bucket
    readAudioFilesFroms3(OUT_FOLDER)
    print('s3 download done')

    # Convert RAW to WAV files
    convertRawToWav(OUT_FOLDER)
    print("----RAW to WAV Conversion Done-----")
