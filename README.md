# RAID

> [!NOTE]\
> RAID : Reinforcement Assisted Image Defense against Adversarial Attacks


## Members
- [Vaughn](https://github.com/webb-c)
- [Suhwan](https://github.com/drrobot333)
- [Hwanhee](https://github.com/khhandrea)
- [UiJin](https://github.com/youuijin)

## Docker execution
git 디렉토리에서 다음 명령어 실행하여 image 생성
```
> docker build -t ku-raid .
```

다음 명령어 실행하여 docker 접속
```
> docker run --rm -it -v (PATH TO DIRECTORY):/volume -p 6006:6006 ku-raid
```

docker 내부에서 다음 명령어 실행하여 코드 작성 (예시)
```
> python main.py
```

docker 내부에서 다음 명령어 실행하여 tensorboard 실행 후, localhost:6006으로 접속
```
> tensorboard --logdir runs --host=0.0.0.0
```