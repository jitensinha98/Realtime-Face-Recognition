import cv2

# using Haar Cascade for face detection
# loading face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capturing video feed from the only webcam the system has
vid = cv2.VideoCapture(0)

# infinite loop
while True : 	

	# returns two parameters - 1.'ret' contains the returned boolean value if the frame is read correctly. 2. 'frame' contains the frame itself
	ret,frame = vid.read()

	# converts the frame from RGB color to gray
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	'''
	we use 'face_cascade.detectMultiScale' to find the faces.
	- '1.3' is the value of >>scaleFactor<< which specifies how much the image size is reduced at each image scale.
           It helps in detection by the algorithm.
        - '5' denotes the >>minNeighbors<< which specifies the parameter specifying how many neighbors 
          each candidate rectangle should have to retain it. This parameter will affect the quality of 
          the detected faces: higher value results in less detections but with higher quality.
        '''
	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)

	# looping on each face with 'x','y' co-ordinate and 'w' & 'h' which denotes width and height respectively.
	for (x,y,w,h) in faces :

		# drawing a rectangle around the face. (0,255,0) denotes color 'green' & '2' is line width.
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

	# cv2.imshow() to display a frame in a window.
	cv2.imshow('Video',frame)

	# this line is used to wait for a key event for each millisecond that should be equal to 'a'.
	if cv2.waitKey(1) & 0xFF == ord('a') :

		# breaks the while loop.
		break;

# closes the camera
vid.release()

# destroys frame window
cv2.destroyAllWindows()

