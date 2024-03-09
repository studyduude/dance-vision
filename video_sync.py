import librosa
import numpy as np
import soundfile as sf
import json
import sys
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array
import os

AUDIO_EXT = ['.m4a','.wav','.mp3','.flac']
RAM_LIM = 1000. # MB, completely arbitrary, just for warning
RAM_WARNING = 'Warning: Using over {0:.1f}GB RAM'.format(RAM_LIM/1e3)

def dtw_shift_param(sig1, sig2, sr):
    """
    Find warping parameters for time shift calculation using Dynamic
    Time Warping (DTW) algorithm from `librosa` package.
    """
    # Code taken from librosa docs
    # Changed metric to 'euclidean', much more robust
    # But Why?

    x_1 = sig1
    x_2 = sig2
    n_fft = int((sr/10.)*2.)
    hop_size = int(n_fft/2.)

    x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=sr, tuning=0,
                                             norm=2, hop_length=hop_size,
                                             n_fft=n_fft)
    x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=sr, tuning=0,
                                             norm=2, hop_length=hop_size,
                                             n_fft=n_fft)

    D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma,
                                 metric='euclidean')
    return (wp, hop_size)

def pseudo_hist_time_shift(wp, sr, hop_size):
    """
    Build Pseudo Histogram to select "mode" of time shift data.

    Most common time shift treated as actual time shift.

    Need proper test to determine confidence in result.
    """
    tdiff_unitless = wp[:,0] - wp[:,1]
    tdiff_unique, tdiff_count = np.unique(tdiff_unitless,
                                          return_counts=True)
    tdiff_sec = tdiff_unique * hop_size / sr

    return (tdiff_sec, tdiff_count)

def find_delay_sec(sig1, sig2, sr):
    """
    Return Time Shift between signals in seconds. Note signals must
    have same sample rate
    """
    # Use Dynamic Time Warping (DTW)
    wp, hop_size = dtw_shift_param(sig1, sig2, sr)

    # Build Pseudo Histogram of time shift "guesses"
    tdiff_sec, tdiff_count = pseudo_hist_time_shift(wp, sr, hop_size)

    # Need a better confidence metric...
    count_argmax = tdiff_count.argmax()

    if count_argmax == 0:
        nearest_argmax_idx = np.array([count_argmax, count_argmax + 1])
    elif count_argmax == len(tdiff_count)-1:
        nearest_argmax_idx = np.array([count_argmax - 1,count_argmax])
    else:
        nearest_argmax_idx = np.array([count_argmax - 1, count_argmax, count_argmax + 1])

    nearest_counts = tdiff_count[nearest_argmax_idx]
    nearest_tdiff = tdiff_sec[nearest_argmax_idx]
    confidence = nearest_counts.sum()/tdiff_count.sum()

    # Weighted average of peak and 2 nearest neighbors
    time_shift = (nearest_tdiff*nearest_counts).sum()/nearest_counts.sum()
    return (time_shift, confidence)

    results_dict[filename]['is_base'] = False
    results_dict[filename]['fake_confidence'] = confidence
    results_dict[filename]['tshift_from_base_sec'] = time_shift


def load_audio(path_to_file):
    file = Path(path_to_file)
    data, sr = librosa.load(str(file.resolve()), sr=None)
    return AudioFile(data, sr, path=file)

def extract_audio(video_path, output_audio_path):
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {output_audio_path}"
    subprocess.run(command, shell=True)

def create_composite_video(json_file, output_video_path):
    # Charger les données du fichier JSON
    with open(json_file, 'r') as f:
        videos_info = json.load(f)
    
    # Initialiser la liste pour stocker les clips vidéo et les offsets
    clips = []
    offsets = []

    # Trouver la vidéo de base et préparer les clips
    for key, video_info in videos_info.items():
        clip = VideoFileClip(video_info['video_path'])
        if video_info['is_base']:
            base_clip = clip
            base_offset = video_info['tshift_from_base_sec']
        else:
            offsets.append(video_info['tshift_from_base_sec'])
            clips.append(clip)

    # Ajuster les clips non-base selon l'offset
    for i, clip in enumerate(clips):
        offset_time = offsets[i] - base_offset
        if offset_time > 0:
            clips[i] = clip.set_start(offset_time)
        else:
            base_clip = base_clip.set_start(-offset_time)

    # Ajouter le clip de base à la liste des clips
    clips.insert(0, base_clip)

    # Créer une vidéo composite
    final_clip = clips_array([clips])

    # Utiliser l'audio de la vidéo de base
    final_clip = final_clip.set_audio(base_clip.audio)

    # Écrire la vidéo de sortie
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")


def files_to_process(video_path_list, audio_path_list):
    audio_files = []
    sr_list = []
    dur_list = []
    results_dict = {}
    tot_MB = 0.
    for i, filepath in enumerate(audio_path_list):
        filepath = Path(filepath)
        if filepath.suffix in AUDIO_EXT:
            file = load_audio(filepath)
            audio_files.append(file)
            filename = file.path.stem
            sr_list.append(file.sr)
            dur_list.append(file.dur_sec)
            tot_MB += file._arr_size_MB
            if tot_MB > RAM_LIM:
                print(RAM_WARNING)
            results_dict[filename] = {'video_path': video_path_list[i],
                                    'audio_path': str(file.path.resolve()),
                                    'sr': file.sr,
                                    'dur_sec': file.dur_sec,
                                    }
        else:
            print('{0} is not supported...'.format(filepath.suffix))
            print('Skipping {0}...'.format(filepath.name))

    print("Files to process:")
    for file in audio_files:
        print('- {0}'.format(file.path.name))
    print('')

    return audio_files, sr_list, dur_list, results_dict

def resample_audio(audio_files, sr_common):
    for file in audio_files:
        if file.sr != sr_common:
            print('For analysis only: resampling {0} from {1} to {2} Hz...'
                    .format(file.path.stem, file.sr, sr_common))
            file.resample(sr_common, inplace=True)

def choose_base_file(audio_files, results_dict, dur_list):
    """
    Choisissez le fichier de base pour la comparaison en fonction de la durée.
    Le fichier le plus long est généralement choisi comme fichier de base.
    
    Parameters:
    audio_files (list): Liste des objets AudioFile chargés.
    dur_list (list): Liste contenant la durée de chaque fichier audio.
    
    Returns:
    AudioFile: Le fichier audio choisi comme base pour la comparaison.
    """
    dur_arr = np.array(dur_list)
    base_file = audio_files[dur_arr.argmax()]
    base_filename = base_file.path.stem
    results_dict[base_filename]['is_base'] = True
    results_dict[base_filename]['tshift_from_base_sec'] = 0.
    print('Using {0} as base sig for comparison...'.format(base_filename))
    return base_file

def save_metadata(results_dict, destination_folder, destination_name):
    """
    Sauvegarde les métadonnées des fichiers audio dans un fichier JSON.

    Parameters:
    results_dict (dict): Dictionnaire contenant les résultats de l'analyse.
    track_folder_name (str): Nom du dossier contenant les fichiers audio analysés.
    """

    # Nom du fichier de métadonnées basé sur le nom du dossier des pistes audio
    print('Writing JSON metadata...')
    results_filename = f"{destination_name}.json"
    json_out_path = Path(destination_folder) / results_filename

    # Écriture des métadonnées dans le fichier JSON
    with open(json_out_path, 'w') as fp:
        json.dump(results_dict, fp, indent=4)

    print(f"Métadonnées sauvegardées dans {json_out_path.resolve()}")
    
    return json_out_path

def compare_to_base(base_file, other_files, sr_common, results_dict):
    for file in other_files:
        filename = file.path.stem
        print("Analyzing {0}...".format(filename))
        sig1 = base_file.data
        sig2 = file.data
        time_shift, confidence = find_delay_sec(sig1, sig2, sr_common)
        if confidence < 0.33: # completely arbitrary cutoff
            print("")
            print("Issue between {0} and {1}".format(base_file, filename))
            print("Are you sure they're recordings from the same session?")
            print("FakeConfidenceMetric = {0:.4f}".format(confidence))
            print("")

        results_dict[filename]['is_base'] = False
        results_dict[filename]['fake_confidence'] = confidence
        results_dict[filename]['tshift_from_base_sec'] = time_shift


class AudioFile:
    def __init__(self, data, sr, path=Path('.')):
        self.path = path
        self.data = data
        self.sr = sr
        self._update_properties()

    def _update_properties(self):
        self.dur_sec = librosa.get_duration(y=self.data, sr=self.sr)
        self._arr_size_MB = self.data.nbytes/1e6 # MB


    def resample(self, sr, inplace = False):
        new_sr = sr
        new_data = librosa.resample(
                        self.data,
                        self.sr,
                        new_sr,
                        res_type = 'kaiser_fast'
                        )
        if inplace:
            self.data = new_data
            self.sr = new_sr
            self._update_properties()
        else:
            resampled_self = AudioFile(new_data, new_sr)
            return resampled_self

def create_metadata(video_path_list, audio_path_list, file_path):
    print("Loading audio files...")
    
    audio_files, sr_list, dur_list, results_dict = files_to_process(video_path_list, audio_path_list)

    # Get "most common" sample rate
    sr_arr = np.array(sr_list)
    sr_uniq, sr_counts = np.unique(sr_arr,return_counts=True)
    sr_common = sr_uniq[sr_counts.argmax()]

    # Resample any signals that are not already "most common" sample rate
    resample_audio(audio_files, sr_common)

    # Get a "base" track to compare all other tracks
    # For now use "longest" track
    base_file = choose_base_file(audio_files,results_dict, dur_list)
    base_filename = base_file.path.stem

    other_files = [file for file in audio_files
                   if file.path.stem != base_filename]

    # Run through all other tracks and compare to "base" track
    compare_to_base(base_file, other_files, sr_common, results_dict)

    print('Finding true base signal from results...')
    tshift_list = [warp_dict['tshift_from_base_sec']
                   for warp_dict in results_dict.values()]
    tshift_arr = np.array(tshift_list)
    true_base_idx = tshift_arr.argmin()
    true_base_name = list(results_dict.keys())[true_base_idx]
    true_offset = np.abs(tshift_arr.min())

    print('Padding audio to align signals...')
    for filekey, warp_dict in results_dict.items():
        sig_path = Path(warp_dict['audio_path'])
        sig_path_str = str(sig_path.resolve())
        sig, sr = librosa.load(sig_path_str, sr=None)
        tshift_orig = warp_dict['tshift_from_base_sec']
        

        if filekey != true_base_name:
            tshift_new = tshift_orig + true_offset
            warp_dict['is_base'] = False
            warp_dict['tshift_from_base_sec'] = tshift_new
            num_zeros = int(tshift_new*sr)
            new_sig = np.concatenate(
                        (np.zeros(num_zeros), sig)
                      ).astype(np.float32)
        else:
            warp_dict['is_base'] = True
            warp_dict['tshift_from_base_sec'] = 0.

    
    json_path = save_metadata(results_dict, file_path, 'metadata')

    return json_path

def extract_audio_from_videos(video_folder, output_folder):
    # Créez le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_path_list = []
    audio_path_list = []
    # Parcourez tous les fichiers du dossier de vidéos
    for video_filename in os.listdir(video_folder):
        if video_filename.endswith(('.mp4', '.mkv', '.avi', '.MP4')):  # Ajoutez d'autres formats si nécessaire
            video_path = os.path.join(video_folder, video_filename)
            output_audio_path = os.path.join(output_folder, video_filename.rsplit('.', 1)[0] + '.wav')
            video_path_list.append(video_path)
            audio_path_list.append(output_audio_path)
            
            # Vérifiez si le fichier audio existe déjà pour éviter de le recréer
            if not os.path.exists(output_audio_path):
                print(f'Extraction de l\'audio de {video_filename}...')
                video_clip = VideoFileClip(video_path)
                video_clip.audio.write_audiofile(output_audio_path)
                video_clip.close()
            else:
                print(f'Le fichier audio {output_audio_path} existe déjà.')

    return video_path_list, audio_path_list

if __name__ == "__main__":
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    final_path = sys.argv[3]
    video_path_list, audio_path_list = extract_audio_from_videos(video_path, audio_path)
    json_path = create_metadata(video_path_list, audio_path_list, final_path)
    create_composite_video(json_path, './choregraphy/chore3/final.mp4')