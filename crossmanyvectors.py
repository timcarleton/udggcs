from numpy import array,matmul
def crossmanyvectors(vectors,vectocross):

    mattocross=array([[0,-vectocross[2],vectocross[1]],[vectocross[2],0,-vectocross[0]],[-vectocross[1],vectocross[0],0]])
    return matmul(vectors,mattocross)
