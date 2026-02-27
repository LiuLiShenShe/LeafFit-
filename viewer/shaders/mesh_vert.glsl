#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform vec3 camera_pos;

out vec3 FragPos;
out vec3 Normal;
out vec3 vertexColor;
out vec2 TexCoord;

void main()
{
    FragPos = position;
    Normal = normal;
    vertexColor = color;
    TexCoord = uv;
    
    gl_Position = projection_matrix * view_matrix * vec4(position, 1.0);
}

