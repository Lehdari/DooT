import os
import time


def main():
    ret = os.system("python3 main.py")
    while ret != 0:
        time.sleep(2)
        ret = os.system("python3 main.py --model model/model")

if __name__ == "__main__":
    main()