import sys
from yolov8s_training import train_model
from yolov8s_validation import validation_model

def print_menu():
    print("\n" + "="*30)
    print("   Eco Scan AI Manager    ")
    print("="*30)
    print(" 1. 모델 학습 시작 (Train)")
    print(" 2. 모델 성능 검증 (Validate)")
    print(" 0. 프로그램 종료 (Exit)")
    print("="*30)

def main():
    while True:
        print_menu()
        user_input = input("실행할 기능을 선택하세요: ").strip()

        if user_input == '1':
            print("\n모델 학습을 시작합니다...")
            try:
                results = train_model()
                print("\n✅ 학습이 성공적으로 완료되었습니다.")
                print(results)
            except Exception as e:
                print(f"\n⛔ 학습 중 오류가 발생했습니다: {e}")

        elif user_input == '2':
            print("\n모델 검증을 시작합니다...")
            try:
                results = validation_model()
                print("\n✅ 검증이 성공적으로 완료되었습니다.")
                print(results)
            except Exception as e:
                print(f"\n⛔ 검증 중 오류가 발생했습니다: {e}")

        elif user_input == '0':
            print("\n프로그램을 종료합니다.")
            sys.exit()

        else:
            print("\n잘못된 입력입니다. 1, 2, 0 중에서 선택해주세요.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n강제 종료되었습니다.")
        sys.exit()