import torchaudio
from tqdm.auto import tqdm
import os

from util import multi_run

class librispeech:

    def __init__(self,
        urls=["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100", "train-clean-360", "train-other-500"],root_path='librispeech/',
        data=None,
        temp_data=None
    ):
        self.urls=urls
        self.root_path=root_path
        self.data=data
        self.temp_data=temp_data

        os.makedirs(self.root_path, exist_ok = True)
    
    def data_iter(self,url=None):
        '''
        return a dict: {'data-set names': (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)} 
        '''
        os.makedirs(self.root_path, exist_ok = True)

        if url: self.urls=url
        self.data = {k:None for k in self.urls}
        for url in self.urls:
            self.data[url] = torchaudio.datasets.LIBRISPEECH(root=self.root_path, url = url, folder_in_archive='LibriSpeech', download= True) 
        return self.data

    def data_process_wav2vec(self,url=None):
        if not self.data: self.data=data_iter(url)

        for url,data in self.data.items():
            self.temp_data = data
            save_path = f"{self.root_path}/{url}/audio"
            os.system(f"rm -rf {save_path}")
            os.makedirs(save_path, exist_ok = True)
            
            data_list = [(url,save_path,k) for k,_ in enumerate(data)]
            trans = multi_run(self.save,data_list,num_workers=36)

            with open(f"{self.root_path}/{url}/transcription.txt", "w") as f:
                f.write("".join(trans))

    def save(self,input_):
        url,save_path,sample_index = input_
        sample = self.temp_data[sample_index]
        file_name = f"{url}_{sample[-2]}_{sample[-3]}_{sample[-1]}"
        path = f"{save_path}/{file_name}.wav"
        torchaudio.save(
            path, sample[0],sample[1],
            encoding="PCM_S", bits_per_sample=64)
        
        return f"{file_name}\t{sample[2].upper()}\n"

if __name__ == "__main__":

    data = librispeech(root_path="/data/part4/librispeech")
    data.data_iter()
    data.data_process_wav2vec()