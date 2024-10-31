def wait_for_saving(waitsec):
    """wait for the file to be written
    Args: 
        waitsec (_type_): int
    """
    import time
    import tqdm
    print("waiting for the file to be written... in ", waitsec, "sec")
    for i in tqdm.tqdm(range(waitsec)):
        time.sleep(1)
