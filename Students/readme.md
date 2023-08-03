# Intrucción para entregar las tareas en el git. 

Si no tiene el fork del directorio del curso:

  1. Crear un fork del directorio del curso Curso_aprendiza_estadistico.
  
  En su respositorio realizar lo siguiente: 
  
  3. Crear una carpeta con su nombre y ultimos digitos del carnet universitarios, crear tambien un archivo readme.md
     Ejemplo: Student/Salinas_431/readme.md
  
  4. Realizar pull request al directorio padre asociado a la cuenta del curso. 
  
Si ya tiene el fork del directorio del curso:
  1. Subir el archivo a su repostitorio del curso.
  2. Hacer pullrequest.

# Guia detallada trabajando con linea de comandos y creando ramas para mejorar el trabajo

- Hacer un fork del repositorio al que se quiere contribuir. Esto creará una copia del repositorio en tu cuenta de GitHub. Para hacer un fork, solo hay que hacer clic en el botón Fork que aparece en la parte superior derecha del repositorio
- Clonar el repositorio forkeado a tu computadora local. Esto te permitirá trabajar con el código de forma local. Para clonar el repositorio, hay que copiar la dirección HTTPS que aparece al hacer clic en el botón Code, y luego ejecutar el comando `git clone [DIRECCIÓN HTTPS]` en la terminal.
- Crear una rama nueva en el repositorio local. Esto te permitirá aislar los cambios que quieras hacer sin afectar a la rama principal. Para crear una rama nueva, hay que ejecutar el comando `git checkout -b [NOMBRE-RAMA]` en la terminal.
- Realizar los cambios que se quieran aportar al repositorio original. Esto puede implicar modificar, añadir o eliminar archivos de código, documentación, etc. Para guardar los cambios, hay que ejecutar los comandos `git add .` para añadir todos los archivos modificados al área de preparación, y `git commit -m "[MENSAJE-COMMIT]"` para crear un commit con un mensaje descriptivo.
- Subir la rama nueva al repositorio remoto forkeado. Esto actualizará el repositorio en tu cuenta de GitHub con los cambios realizados. Para subir la rama, hay que ejecutar el comando `git push origin [NOMBRE-RAMA]` en la terminal.
- Crear un pull request desde el repositorio remoto forkeado al repositorio original. Esto solicitará al dueño o a los colaboradores del repositorio original que revisen y aprueben los cambios propuestos. Para crear un pull request, hay que ir a la página del repositorio forkeado en GitHub, seleccionar la rama nueva y hacer clic en el botón Compare & pull request. Luego, hay que escribir un título y una descripción para el pull request, y hacer clic en Create pull request.
- Esperar a que el pull request sea revisado y aceptado o rechazado por el dueño o los colaboradores del repositorio original. También se pueden recibir comentarios o sugerencias para mejorar el pull request. En ese caso, se pueden hacer más cambios en la rama local y volver a subirla al repositorio remoto forkeado para actualizar el pull request.
