#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec3 vertexColor;
in vec2 TexCoord;

uniform vec3 camera_pos;
uniform sampler2D texture_diffuse;
uniform bool has_texture;

out vec4 FragColor;

void main()
{
    vec3 baseColor;
    if (has_texture) {
        baseColor = texture(texture_diffuse, TexCoord).rgb;
    } else {
        baseColor = vertexColor;
    }
    
    vec3 lightPos = camera_pos; 
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    
    float diff = abs(dot(norm, lightDir));
    vec3 diffuse = diff * lightColor;
    
    float specularStrength = 0.5;
    vec3 viewDir = normalize(camera_pos - FragPos);
    
    vec3 adjustedNorm = dot(norm, viewDir) < 0.0 ? -norm : norm;
    vec3 reflectDir = reflect(-lightDir, adjustedNorm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    vec3 result = (ambient + diffuse + specular) * baseColor;
    FragColor = vec4(result, 1.0);
}

