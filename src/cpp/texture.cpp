#define STB_IMAGE_IMPLEMENTATION
#include "../hpp/texture.hpp"

// Call this during cleanup to prevent memory leaks
TextureManager::~TextureManager()
{
    for (auto &tex : textures)
    {
        if (tex.data)
        {
            stbi_image_free(tex.data);
        }
    }
}

// Thread-safe / Duplication-safe loading function
int TextureManager::load_texture(const std::string &filepath)
{
    // If the texture was already loaded previously, return its existing ID instantly
    if (filename_to_id.find(filepath) != filename_to_id.end())
    {
        return filename_to_id[filepath];
    }

    // Otherwise, read it from disk for the first time
    TextureImage tex;
    tex.filename = filepath;
    tex.data = stbi_load(filepath.c_str(), &tex.width, &tex.height, &tex.channels, 3);

    if (!tex.data)
    {
        std::cerr << "WARNING: Failed to load texture image: " << filepath << std::endl;
        return -1; // Invalid texture indicator
    }

    // Add to array and register its index position
    int new_id = static_cast<int>(textures.size());
    textures.push_back(tex);
    filename_to_id[filepath] = new_id;

    std::cout << "Successfully cached texture [" << new_id << "]: " << filepath << std::endl;
    return new_id;
}

// 1. Update your texture sampling lookup to pick from the cached array
color TextureManager::sample_texture_from_cache(const TextureManager &tex_manager, int tex_id, float u, float v)
{
    if (tex_id < 0 || tex_id >= static_cast<int>(tex_manager.textures.size()))
    {
        return color(1, 0, 1); // Return bright magenta to easily spot a broken texture ID bug
    }

    const TextureImage &tex = tex_manager.textures[tex_id];

    // Call your existing sample_texture function using this specific image's metrics
    return sample_texture(tex.data, tex.width, tex.height, u, v);
}