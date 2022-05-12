from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

def multi_run(func, data_list,num_workers=1):
    print("---Running multi-processing---")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm((executor.map(func, data_list)), total=len(data_list)))
    
    return results