import numpy as np
#주석은 c++로 대체했을 때 상황 및 설명

x = np.random.rand(3,2)
#int x = x[3][2]

print(x)

# : 이 의미하는것 == '모든'
# 즉 모든 행에서 0번째 값을 뽑아 오라는 코드는 아래와 같다
x = x[:, 0]
# for (i = 0, i < 3, i++){
#   y[i] = x[i][2]
# }


print(x)
#cout << y