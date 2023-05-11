

# A function for transfering image to numpy array 
def create_all_images(idx):
    row = df.iloc[idx]
    im = []
    for i in range(1,6):
        local_im = cv2.imread(base_dir + row.plate + '/' + row['C' + str(i)], -1)
        dmso_mean = dmso_stats_df[row.plate]['C' + str(i)]['m']
        dmso_std = dmso_stats_df[row.plate]['C' + str(i)]['std']
        local_im = dmso_normalization(local_im, dmso_mean, dmso_std)

        im.append(local_im)
    im = np.array(im).transpose(1, 2, 0).astype("float")
    im = np.array(easy_transforms(image = im)['image'])

    return im

# Write all the images into a big numpy array  
all_images = np.zeros((df.shape[0], 256, 256, 5), dtype = np.float32)
for f in range(df.shape[0]):
    all_images[f] = create_all_images(f)

# Save the big numpy array 
with open('all_images.npy', 'wb') as f:
    np.save(f, all_images)

