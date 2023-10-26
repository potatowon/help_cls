import logging

def setting_logger(file_name : str):
    """
    주어진 파일 이름을 사용하여 로거를 설정하고 반환합니다.
    
    로그 메시지는 주어진 파일과 콘솔에 동시에 기록됩니다. 파일 로그에는 메시지의 발생 시간이 포함되며, 
    콘솔 로그에는 메시지의 로그 레벨과 내용만 포함됩니다.
    
    Args:
        file_name (str): 로그 메시지를 저장할 파일의 이름

    Returns:
        logging.Logger: 설정된 로거 객체.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # 파일 핸들러 생성 및 설정
    file_handler = logging.FileHandler(file_name)
    # file_handler.setLevel(logging.INFO) # file에는 출력되는 값보다 더 적게 기록
    file_format = logging.Formatter('%(asctime)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # 콘솔 핸들러 생성 및 설정
    stream_handler = logging.StreamHandler()
    stream_format = logging.Formatter('%(levelname)s: %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)
    
    return logger

