# robot20
# aula 24/03

Usando o simulador Gazebo e o robô Turtlebot3.

Antes de começar, precisamos atualizar nosso repositorio my_simulation, para isso abra um terminal Crtl+Alt+t e digite:

    cd ~/catkin_ws/src/my_simulation
    git pull  

Execute o comando catkin_make na pasta raiz catkin_ws
    
    cd ~/catkin_ws
    catkin_make
Execute o comando catkin_make na pasta raiz catkin_ws
    
    cd ~/catkin_ws
    catkin_make    

Vamos subir o nosso mundo simulado
    
    roslaunch my_simulation sala404_creepers.launch
    
Abra um novo terminal (Crtl+Alt+t) e execute o rosrun para renomear o topico da câmera
    
    rosrun topic_tools relay  /camera/rgb/image_raw/compressed /kamera

Abra um novo terminal (Crtl+Alt+t) e execute o rosrun para renomear o topico da câmera
    
    rosrun exemplos_python cor.py
    
Se tudo ocorreu bem, agora o nosso turtlebot esta realizado a segmentação de cor e detectando a caixa vermelho em nosso mundo simulado.
    
## Video do guia
Fiquei com duvida no guia, acompanha o video.

https://www.youtube.com/watch?v=OKphJFyhnSg&feature=youtu.be


Se der alguma coisa errada, consulte esses guia de sobrevivência ROS Gazebo

https://www.youtube.com/playlist?list=PLM8rZg4fCalht-rexa91MO1y3jxm9mwOa
