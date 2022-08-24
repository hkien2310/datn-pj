from djitellopy import tello


#Khai báo biến
datn = tello.Tello()
# Kết nối với drone
datn.connect()

#In ra giá trị của pin
print(datn.get_battery())