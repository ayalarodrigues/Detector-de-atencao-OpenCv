#O shape_predictor_68_face_landmarks.dat representa 68 pontos predefinidos no rosto usados para rastrear os olhos
#A distância euclidiana é usada para verificar a movimentação dos olhos

import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib


OLHO_LIM = 0.3 #limite dos olhos
OLHO_FRAMES = 7 #frames dos olhos 
INFO = True

mpo = 0 #média da proporção entre os olhos


CONTADOR_FRAMES = 0 #contador de frames dos olhos
CONTADOR_MOV = 0      #contador de movimentação ocular


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)



camera = cv2.VideoCapture(0)
ret, frame = camera.read()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


#modelo em 3D
model_points = np.array([(0.0, 0.0, 0.0), #nariz
                         (0.0, -330.0, -65.0), #queixo        
                         (-225.0, 170.0, -135.0), #canto esquerdo - olho esquerdo   
                         (225.0, 170.0, -135.0),  #canto direito - olho direito   
                         (-150.0, -150.0, -125.0),#canto esquerdo da boca    
                         (150.0, -150.0, -125.0)]) #canto direito da boca

focal_length = frame.shape[1]
center = (frame.shape[1]/2, frame.shape[0]/2)

camera_matrix = np.array([[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double")

dist_coeffs = np.zeros((4,1)) #supõe-se que não há distorção na lente

t_end = time.time()
while(True):
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)


        #extrai as coordenadas do olho esquerdo e direito e, 
        #em seguida, usa as coordenadas para calcular 
        #a proporção dos dois olhos
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye) 
        mpo = (leftEAR + rightEAR) / 2.0
        

        image_points = np.array([
                                (shape[30][0], shape[30][1]),
                                (shape[8][0], shape[8][1]),
                                (shape[36][0], shape[36][1]),
                                (shape[45][0], shape[45][1]),
                                (shape[48][0], shape[48][1]),
                                (shape[54][0], shape[54][1])
                                ], dtype="double")


        #Localiza uma pose de objeto a partir das correspondências de pontos 3D-2D.
        #cv2.solvePnP( objectPoints, imagePoints, cameraMatrix, distCoeffs [ , rvec [ , TVEC [ , useExtrinsicGuess [ , flags ] ] ] ] )
        #objectPoints - Matriz de pontos do objeto no espaço de coordenadas do objeto
        #imagePoints - Matriz dos pontos de imagem correspondentes
        #cameraMatrix - matriz da câmera de entrada
        #distCoeffs - vetor de entrada dos coeficientes de distorção
        #ITERATIVE - Nesse caso, a função encontra uma pose que minimiza o erro de reprojeção, ou seja, a soma das distâncias ao quadrado entre as projeções observadas e as projetadas
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        #Projeta um ponto 3D (0.0, 0.0, 1000.0) no plano da imagem e desenha uma linha saindo do nariz
        #jacobian - Matriz jacobiana de saída opcional, 3x9 ou 9x3, que é uma matriz de derivadas parciais dos componentes da matriz de saída em relação aos componentes da matriz de entrada.

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)


        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        #pi e p2 são vetores que definem a posição da cabeça

        

        if CONTADOR_MOV > 30:
            cv2.putText(frame, "Perda de atencao!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            


        if mpo < OLHO_LIM:
            CONTADOR_FRAMES += 1

            if CONTADOR_FRAMES >= OLHO_FRAMES:
                cv2.putText(frame, "Desatento!", (200, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if CONTADOR_FRAMES > 2:
                CONTADOR_MOV += 1
            CONTADOR_FRAMES = 0
        
        
        
        if (time.time() - t_end) > 60:
            t_end = time.time()
            CONTADOR_MOV = 0

        

    if INFO:
        cv2.putText(frame, "Movimentos: {}".format(CONTADOR_MOV), (10, 30),
            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)


    cv2.imshow("Detector de atencao", frame)
    if cv2.waitKey(1) == ord('q'):
        break

     
  
    time.sleep(0.02)
    
camera.release()  
cv2.destroyAllWindows()

