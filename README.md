# 휴먼로보틱스 / 고급로봇공학 2026

본 저장소는 **휴먼로보틱스** / **고급로봇공학** 강의의 과제 및 프로젝트를 위한 레포지토리입니다.  
본 저장소에는 **연세대학교 신동준 교수님**과 **서울과학기술대학교 김정엽 교수님**의 공동 강의 자료 및 코드가 포함되어 있습니다.

## 업데이트 예정

| 항목 | 업데이트 예정일 |
|------|-----------------|
| HW1 | 4월 01일 |
| HW2 | 4월 27일 |
| HW3 | 5월 11일 |
| 프로젝트 | 5월 25일 |

---

## 설치 가이드

### 1. 저장소 준비

**방법 A**

```bash
git clone https://github.com/hcair/HumanRobotics-AdvancedRobotics.git
cd HumanRobotics-AdvancedRobotics
```

**방법 B**

- GitHub 저장소 페이지에서 **Code → Download ZIP**을 선택하여 다운로드
- 압축 해제 후 생성된 저장소 폴더로 이동

### 2. Python 버전 확인

Python **3.10, 3.11** 버전을 권장합니다.  
테스트 환경은 3.10/3.11이며, **3.12 이상에서는 설치 실패 가능성**이 있습니다.

```bash
python --version
# 또는
python3 --version
```

### 3. 가상환경 생성 및 활성화

**Windows (CMD):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

PowerShell에서 **실행 정책** 관련 오류가 발생하는 경우 아래 명령을 실행합니다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. 의존성 설치

가상환경을 활성화한 뒤 프로젝트 루트에서 실행합니다.

**Windows (CMD / PowerShell)**

```cmd
python.exe -m pip install --upgrade pip
python.exe -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

---

## 프로젝트 구조

| 폴더 | 설명 |
|------|------|
| `core/` | 공통 모듈 (로봇 래퍼, 실시간 동기화, 키보드/입력, GUI 등) |
| `homeworks/` | 과제 (hw1, hw2, hw3) — 각 폴더에 과제별 설명 및 스크립트 |
| `models/` | MuJoCo XML 모델 (로봇/환경 등) |
| `project/` | 프로젝트용 코드 및 자료 (추후 업데이트 예정) |

과제 및 프로젝트는 위 구조를 기준으로 순차적으로 업데이트됩니다.

---

## 실행 방법

아래 명령은 프로젝트 루트 디렉터리에서 실행합니다.

**HW1 예시 (Open Manipulator X):**

**Windows (PowerShell / CMD):**

```powershell
python homeworks/hw1/hw1_open_manipulator.py
```

**Linux / macOS:**

```bash
python3 homeworks/hw1/hw1_open_manipulator.py
```

---

## 키보드·마우스

### 키보드 (앱 공통)

| 키 | 동작 |
|----|------|
| **Space** | 일시정지 / 재개 |
| **R** | 리셋 |
| **Q** | 종료 |
| **+** / **-** | 시뮬레이션 속도 증가 / 감소 |

### 마우스 (MuJoCo 뷰어)

| 동작 | 설명 |
|------|------|
| **더블클릭** | Body 선택 |
| **Ctrl + 왼쪽 드래그** | 선택한 body에 외력 가하기 |
| **스크롤** | 줌 |
| **왼쪽 드래그** | 시점 회전 |
| **Shift + 오른쪽 드래그** | 시점 팬 |
