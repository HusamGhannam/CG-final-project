#include <iostream>
#include <vector>
#include <cmath>

// Main definitions
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ==============================================================================
// CUSTOM 3D MATH LIBRARY 
// ==============================================================================
const float PI = 3.14159265359f;

struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
};

struct Mat4 {
    float m[16]; // Row-major 4x4 matrix

    // Identity Matrix Constructor
    Mat4() {
        for (int i = 0; i < 16; i++) m[i] = 0.0f;
        m[0] = 1.0f; m[5] = 1.0f; m[10] = 1.0f; m[15] = 1.0f;
    }

    // Matrix Multiplication (Right-to-Left logic)
    static Mat4 multiply(const Mat4& a, const Mat4& b) {
        Mat4 res;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                res.m[i * 4 + j] = 0;
                for (int k = 0; k < 4; k++)
                    res.m[i * 4 + j] += a.m[i * 4 + k] * b.m[k * 4 + j];
            }
        }
        return res;
    }

    // Translation Matrix
    static Mat4 translate(float x, float y, float z) {
        Mat4 res;
        res.m[3] = x; res.m[7] = y; res.m[11] = z;
        return res;
    }

    // Scaling Matrix
    static Mat4 scale(float x, float y, float z) {
        Mat4 res;
        res.m[0] = x; res.m[5] = y; res.m[10] = z;
        return res;
    }

    // Rotation Matrix (X-Axis for rolling the ball)
    static Mat4 rotateX(float angle) {
        Mat4 res;
        float c = cos(angle), s = sin(angle);
        res.m[5] = c;  res.m[6] = -s;
        res.m[9] = s;  res.m[10] = c;
        return res;
    }

    // Perspective Projection (The Lens)
    static Mat4 perspective(float fovRad, float aspect, float nearPlane, float farPlane) {
        Mat4 res;
        for (int i = 0; i < 16; i++) res.m[i] = 0.0f;
        float f = 1.0f / tan(fovRad / 2.0f);
        res.m[0] = f / aspect;
        res.m[5] = f;
        res.m[10] = (farPlane + nearPlane) / (nearPlane - farPlane);
        res.m[11] = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
        res.m[14] = -1.0f;
        return res;
    }
};

// ==============================================================================
// SHADER STRINGS [cite: 968, 1319]
// ==============================================================================
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 aNormal;\n"
"layout (location = 2) in vec2 aTexCoords;\n"
"out vec3 FragPos;\n"
"out vec3 Normal;\n"
"out vec2 TexCoords;\n"
"uniform mat4 model;\n"
"uniform mat4 view;\n"
"uniform mat4 projection;\n"
"void main() {\n"
"   vec4 worldPos = model * vec4(aPos, 1.0);\n"
"   gl_Position = projection * view * worldPos;\n"
"   FragPos = vec3(worldPos);\n"
"   // Normal matrix approximation (ignores non-uniform scaling)\n"
"   Normal = mat3(model) * aNormal;\n"
"   TexCoords = aTexCoords;\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"in vec3 FragPos;\n"
"in vec3 Normal;\n"
"in vec2 TexCoords;\n"
"uniform sampler2D diffuseTex;\n"
"uniform vec3 lightPos;\n"
"void main() {\n"
"   // Basic Phong Lighting (Ambient + Diffuse)\n"
"   vec3 lightColor = vec3(1.0, 1.0, 1.0);\n"
"   float ambientStrength = 0.4;\n"
"   vec3 ambient = ambientStrength * lightColor;\n"
"   vec3 norm = normalize(Normal);\n"
"   vec3 lightDir = normalize(lightPos - FragPos);\n"
"   float diff = max(dot(norm, lightDir), 0.0);\n"
"   vec3 diffuse = diff * lightColor;\n"
"   vec4 texColor = texture(diffuseTex, TexCoords);\n"
"   vec3 result = (ambient + diffuse) * texColor.rgb;\n"
"   FragColor = vec4(result, texColor.a);\n"
"}\0";

// ==============================================================================
// GEOMETRY GENERATORS & HELPERS
// ==============================================================================

// Helper to load textures [cite: 140, 150]
unsigned int loadTexture(const char* path) {
    unsigned int textureID;
    glGenTextures(1, &textureID);
    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path, &width, &height, &nrChannels, 0);
    if (data) {
        GLenum format = (nrChannels == 4) ? GL_RGBA : GL_RGB;
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(data);
    }
    else {
        std::cout << "Failed to load texture: " << path << std::endl;
    }
    return textureID;
}

// Generates a proper Sphere (Vertices, Normals, UVs)
void generateSphere(float radius, int sectors, int stacks, std::vector<float>& vertices, std::vector<unsigned int>& indices) {
    float x, y, z, xy, nx, ny, nz, s, t;
    float sectorStep = 2 * PI / sectors;
    float stackStep = PI / stacks;

    for (int i = 0; i <= stacks; ++i) {
        float stackAngle = PI / 2 - i * stackStep;
        xy = radius * cos(stackAngle);
        z = radius * sin(stackAngle);

        for (int j = 0; j <= sectors; ++j) {
            float sectorAngle = j * sectorStep;
            x = xy * cos(sectorAngle);
            y = xy * sin(sectorAngle);
            nx = x / radius; ny = y / radius; nz = z / radius;
            s = (float)j / sectors; t = (float)i / stacks;

            vertices.push_back(x); vertices.push_back(y); vertices.push_back(z); // Pos
            vertices.push_back(nx); vertices.push_back(ny); vertices.push_back(nz); // Normal
            vertices.push_back(s); vertices.push_back(t); // UV
        }
    }

    for (int i = 0; i < stacks; ++i) {
        int k1 = i * (sectors + 1);
        int k2 = k1 + sectors + 1;
        for (int j = 0; j < sectors; ++j, ++k1, ++k2) {
            if (i != 0) { indices.push_back(k1); indices.push_back(k2); indices.push_back(k1 + 1); }
            if (i != (stacks - 1)) { indices.push_back(k1 + 1); indices.push_back(k2); indices.push_back(k2 + 1); }
        }
    }
}

// ==============================================================================
// MAIN GAME ENGINE
// ==============================================================================

struct Tile { float x, z; bool isHole; };

int main() {
    // 1. Core Initialization [cite: 854]
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "Rolling Ball - Advanced 3D", NULL, NULL);
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    glewInit();

    glEnable(GL_DEPTH_TEST);

    // 2. Compile Shaders
    unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSource, NULL);
    glCompileShader(vs);
    unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSource, NULL);
    glCompileShader(fs);
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vs);
    glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram);

    // 3. Load Textures
    unsigned int ballTex = loadTexture("assets/ball.png");
    unsigned int floorTex = loadTexture("assets/wood.png");

    // 4. Setup Geometry: SPHERE
    std::vector<float> sphereVerts;
    std::vector<unsigned int> sphereIdx;
    generateSphere(0.5f, 36, 18, sphereVerts, sphereIdx);

    unsigned int ballVAO, ballVBO, ballEBO;
    glGenVertexArrays(1, &ballVAO); glGenBuffers(1, &ballVBO); glGenBuffers(1, &ballEBO);
    glBindVertexArray(ballVAO);
    glBindBuffer(GL_ARRAY_BUFFER, ballVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVerts.size() * sizeof(float), sphereVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ballEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIdx.size() * sizeof(unsigned int), sphereIdx.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float))); glEnableVertexAttribArray(2);

    // 5. Setup Geometry: FLOOR CUBE
    float cubeVerts[] = {
        // Top Face (Used for floor surface)
        -1.0f, 0.0f, -1.0f,  0.0f, 1.0f, 0.0f,  0.0f, 0.0f,
         1.0f, 0.0f, -1.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,
         1.0f, 0.0f,  1.0f,  0.0f, 1.0f, 0.0f,  1.0f, 1.0f,
        -1.0f, 0.0f,  1.0f,  0.0f, 1.0f, 0.0f,  0.0f, 1.0f,
    };
    unsigned int cubeIdx[] = { 0, 1, 2, 2, 3, 0 };

    unsigned int cubeVAO, cubeVBO, cubeEBO;
    glGenVertexArrays(1, &cubeVAO); glGenBuffers(1, &cubeVBO); glGenBuffers(1, &cubeEBO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVerts), cubeVerts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cubeIdx), cubeIdx, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))); glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float))); glEnableVertexAttribArray(2);

    // 6. Game State Variables
    float ballX = 0.0f, ballY = 0.5f, ballZ = 0.0f;
    float velocityY = 0.0f;
    float gameSpeed = 8.0f;
    bool isJumping = false;
    bool isFalling = false;
    float distanceTraveled = 0.0f;
    const float GRAVITY = -15.0f;

    // Procedural Track Generation
    std::vector<Tile> track;
    for (int i = 0; i < 20; i++) {
        // Create 3 lanes (Left, Center, Right)
        track.push_back({ -2.0f, -i * 2.0f, false });
        track.push_back({ 0.0f, -i * 2.0f, false });
        track.push_back({ 2.0f, -i * 2.0f, false });
    }

    float lastTime = glfwGetTime();

    // ==============================================================================
    // RENDER LOOP
    // ==============================================================================
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        // --- PHYSICS & INPUT ---
        if (!isFalling) {
            // Speed accelerates over time
            gameSpeed += 0.1f * deltaTime;
            ballZ -= gameSpeed * deltaTime;
            distanceTraveled += gameSpeed * deltaTime;

            // Lateral Movement (A/D)
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) ballX -= 6.0f * deltaTime;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) ballX += 6.0f * deltaTime;

            // Jump Logic
            if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !isJumping) {
                velocityY = 6.0f;
                isJumping = true;
            }
        }

        // Apply Gravity (Y-Axis)
        if (isJumping || isFalling) {
            velocityY += GRAVITY * deltaTime;
            ballY += velocityY * deltaTime;
        }

        // --- COLLISION DETECTION (Sphere to AABB Floor) ---
        bool groundedThisFrame = false;

        // Dynamic Track Generation
        if (track[0].z > ballZ + 5.0f) {
            // Remove tiles passed by camera
            track.erase(track.begin(), track.begin() + 3);

            // Generate new row far ahead. 10% chance to be a "Hole"
            float newZ = track.back().z - 2.0f;
            track.push_back({ -2.0f, newZ, (rand() % 10 == 0) });
            track.push_back({ 0.0f, newZ, (rand() % 10 == 0) });
            track.push_back({ 2.0f, newZ, (rand() % 10 == 0) });
        }

        // Check if ball is over an active tile
        for (auto& tile : track) {
            if (tile.isHole) continue;
            // AABB Bounds of the tile (Tile size is 2x2)
            if (ballX > tile.x - 1.0f && ballX < tile.x + 1.0f &&
                ballZ > tile.z - 1.0f && ballZ < tile.z + 1.0f) {
                groundedThisFrame = true;
                break;
            }
        }

        // Resolve Y-axis state
        if (!isFalling) {
            if (groundedThisFrame && ballY <= 0.5f) {
                ballY = 0.5f; // Clamp to floor
                isJumping = false;
                velocityY = 0.0f;
            }
            else if (!groundedThisFrame && ballY <= 0.5f) {
                // Ball fell into a hole!
                isFalling = true;
            }
        }

        // Game Over trigger
        if (ballY < -10.0f) {
            std::cout << "GAME OVER! Final Score (Distance): " << (int)distanceTraveled << std::endl;
            glfwSetWindowShouldClose(window, true);
        }

        // --- RENDERING ---
        glClearColor(0.4f, 0.7f, 1.0f, 1.0f); // Sky Blue
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);// [cite:1292, 1293]

            glUseProgram(shaderProgram);

        // Send Light Position (Following the ball)
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), ballX, 5.0f, ballZ);

        // View & Projection Matrices
        // Camera stays behind and slightly above the ball
        Mat4 view = Mat4::translate(-ballX, -2.0f, -(ballZ + 6.0f));
        Mat4 proj = Mat4::perspective(45.0f * (PI / 180.0f), 800.0f / 600.0f, 0.1f, 100.0f);

        // Note: Using GL_TRUE to transpose our Row-Major matrices for OpenGL
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_TRUE, view.m);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_TRUE, proj.m);

        // 1. Draw Track
        glBindVertexArray(cubeVAO);
        glBindTexture(GL_TEXTURE_2D, floorTex);
        for (auto& tile : track) {
            if (tile.isHole) continue;
            Mat4 modelTile = Mat4::translate(tile.x, 0.0f, tile.z);
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_TRUE, modelTile.m);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }

        // 2. Draw Ball
        glBindVertexArray(ballVAO);
        glBindTexture(GL_TEXTURE_2D, ballTex);

        // Calculate Rolling Rotation
        // Circumference = 2 * PI * r. Rotation = Distance / Radius
        float rotationAngle = distanceTraveled / 0.5f;

        Mat4 modelBallRot = Mat4::rotateX(-rotationAngle);
        Mat4 modelBallTrans = Mat4::translate(ballX, ballY, ballZ);
        Mat4 modelBall = Mat4::multiply(modelBallTrans, modelBallRot);

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_TRUE, modelBall.m);
        glDrawElements(GL_TRIANGLES, sphereIdx.size(), GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}
