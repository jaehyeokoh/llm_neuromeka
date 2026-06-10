from kokoro import KPipeline
import torch
import threading
import re
import time
import contextlib
import sounddevice as sd
import numpy as np

voice="af_bella"


@contextlib.contextmanager
def torch_thread_limit(num_threads):
    """PyTorch 스레드 수를 임시로 변경하는 컨텍스트 매니저"""
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(num_threads)
        yield
    finally:
        torch.set_num_threads(original_threads)

class KokoroEnglishTTS:
    def __init__(self):
        self.pipeline = None
        self._initialization_task = None
        self._is_initialized = False
        self.tts_thread_count = 5  # 스레드 수 줄임 (6→4)
        self.playback_speed = 1.01  # 재생 속도 (1.0 = 정상, 1.5 = 1.5배속, 0.8 = 0.8배속)
        
        # sounddevice 기본 설정 (ALSA 최적화)
        sd.default.blocksize = 2048
        sd.default.latency = 'low'
        sd.default.dtype = 'float32'
    
    def initialize_pipeline(self, lang_code: str = 'a') -> bool:
        """기존 동기 초기화 (호환성 유지) - 메인 코드에 영향 없음"""
        try:
            print("모델 초기화 중...")
            # 초기화 시에만 임시로 스레드 수 변경
            with torch_thread_limit(self.tts_thread_count):
                self.pipeline = KPipeline(lang_code=lang_code, device="cpu") # 일단 cpu로 함
            print("초기화 완료")
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"초기화 실패: {e}")
            self._is_initialized = False
            return False

    def start_initialization(self, lang_code: str = 'a'):
        """백그라운드에서 초기화 시작 (fire-and-forget)"""
        if self._initialization_task is not None and self._initialization_task.is_alive(): 
            print("이미 초기화가 진행 중입니다.")
            return
            
        def _background_init():
            self.initialize_pipeline(lang_code)
            
        self._initialization_task = threading.Thread(target=_background_init, daemon=True)
        self._initialization_task.start()
        print("모델 초기화가 백그라운드에서 시작되었습니다.")

    def wait_for_initialization(self, timeout: float = 10.0):
        start_time = time.time()
        while not self._is_initialized:
            if time.time() - start_time > timeout:
                raise TimeoutError("모델 초기화 시간 초과")
            time.sleep(0.1)

    def _generate_audio(self, text: str, voice = voice):
        """음성 생성 시에도 스레드 제한 적용 - 메인 코드에 영향 없음"""
        if not self._is_initialized:
            print("모델이 준비되지 않았습니다. 초기화를 기다립니다...")
            self.wait_for_initialization()
    
        try:
            # 음성 생성 시에만 임시로 스레드 수 변경
            with torch_thread_limit(self.tts_thread_count):
                generator = self.pipeline(text, voice=voice)
                for _, _, audio in generator:
                    return audio, 24000
        except Exception as e:
            print(f"음성 생성 실패: {e}")
            return None

    def _play_audio(self, audio_data, sample_rate, done_event=None):
        """스트리밍 재생으로 변경 - 파일 I/O 없음, 속도 조절 가능, ALSA 최적화"""
        def _play():
            try:
                # numpy array로 변환
                if not isinstance(audio_data, np.ndarray):
                    audio_data_np = np.array(audio_data, dtype=np.float32)
                else:
                    audio_data_np = audio_data.astype(np.float32)
                
                # 속도 조절 적용
                adjusted_sample_rate = int(sample_rate * self.playback_speed)
                
                # ALSA 최적화: 큰 버퍼, 적은 지연시간
                sd.play(
                    audio_data_np, 
                    adjusted_sample_rate,
                    blocksize=2048,  # 버퍼 크기 증가
                    latency='low'    # 지연시간 설정
                )
                sd.wait()  # 재생 완료까지 대기
                
            except Exception as e:
                print(f"재생 실패: {e}")
            finally:
                if done_event:
                    done_event.set()
        
        # 스레드만 시작하고 바로 반환
        thread = threading.Thread(target=_play, daemon=False)
        thread.start()
        return True  # 즉시 반환

    def speak_auto_sequential(self, text) -> threading.Event:
        done = threading.Event()
        
        if not isinstance(text, str):
            text = str(text)
        
        def _run():
            local_text = text
            local_text = re.sub(r'[^a-zA-Z\s.!?]', '', local_text)
            segments = [seg.strip() for seg in re.split(r'[.!?]', local_text) if seg.strip()]
            
            # 모든 작업을 백그라운드 스레드에서 처리
            def _generate_and_play():
                for segment in segments:
                    result = self._generate_audio(segment, voice)
                    if result:
                        audio_data, sr = result
                        # 각 세그먼트 재생이 끝날 때까지 기다림 (순차 재생 보장)
                        segment_done = threading.Event()
                        self._play_audio_with_callback(audio_data, sr, segment_done)
                        segment_done.wait()  # 이 세그먼트 재생 완료까지 대기
                
                done.set()  # 모든 재생 완료
            
            threading.Thread(target=_generate_and_play, daemon=False).start()
        
        threading.Thread(target=_run, daemon=False).start()
        return done  # 즉시 반환, 메인 코드는 블로킹되지 않음

    def set_playback_speed(self, speed: float):
        """재생 속도 설정
        Args:
            speed: 재생 속도 (1.0 = 정상, 1.5 = 1.5배속, 0.8 = 0.8배속)
        """
        if speed <= 0:
            raise ValueError("재생 속도는 0보다 커야 합니다")
        self.playback_speed = speed
        print(f"재생 속도를 {speed}배로 설정했습니다")

    def _play_audio_with_callback(self, audio_data, sample_rate, done_event):
        """스트리밍 재생으로 변경 - 재생 완료시 콜백 호출, 속도 조절 가능, ALSA 최적화"""
        def _play():
            try:
                # numpy array로 변환
                if not isinstance(audio_data, np.ndarray):
                    audio_data_np = np.array(audio_data, dtype=np.float32)
                else:
                    audio_data_np = audio_data.astype(np.float32)
                
                # 속도 조절 적용
                adjusted_sample_rate = int(sample_rate * self.playback_speed)
                
                # ALSA 최적화: 큰 버퍼, 적은 지연시간
                sd.play(
                    audio_data_np, 
                    adjusted_sample_rate,
                    blocksize=2048,  # 버퍼 크기 증가
                    latency='low'    # 지연시간 설정
                )
                sd.wait()  # 재생 완료까지 대기
                
            except Exception as e:
                print(f"재생 실패: {e}")
            finally:
                done_event.set()  # 재생 완료 신호
        
        threading.Thread(target=_play, daemon=False).start()