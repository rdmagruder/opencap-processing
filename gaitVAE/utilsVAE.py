import numpy as np
from scipy.spatial.distance import mahalanobis
import torch
import torch.nn as nn

VAE_COLUMN_ORDER = [
    'time',
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    'hip_flexion_ips', 'hip_adduction_ips', 'hip_rotation_ips',
    'knee_angle_ips', 'ankle_angle_ips', 'subtalar_angle_ips',
    'hip_flexion_contra', 'hip_adduction_contra', 'hip_rotation_contra',
    'knee_angle_contra', 'ankle_angle_contra', 'subtalar_angle_contra',
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
    'arm_add_ips', 'arm_flex_ips', 'arm_rot_ips', 'elbow_flex_ips', 'pro_sup_ips',
    'arm_add_contra', 'arm_flex_contra', 'arm_rot_contra', 'elbow_flex_contra', 'pro_sup_contra',
]

def get_new_col_names(leg):
    if leg == 'r':
        return {
            'hip_flexion_r': 'hip_flexion_ips',
            'hip_adduction_r': 'hip_adduction_ips',
            'hip_rotation_r': 'hip_rotation_ips',
            'knee_angle_r': 'knee_angle_ips',
            'ankle_angle_r': 'ankle_angle_ips',
            'subtalar_angle_r': 'subtalar_angle_ips',
            'mtp_angle_r': 'mtp_angle_ips',
            'hip_flexion_l': 'hip_flexion_contra',
            'hip_adduction_l': 'hip_adduction_contra',
            'hip_rotation_l': 'hip_rotation_contra',
            'knee_angle_l': 'knee_angle_contra',
            'ankle_angle_l': 'ankle_angle_contra',
            'subtalar_angle_l': 'subtalar_angle_contra',
            'mtp_angle_l': 'mtp_angle_contra',

            'arm_add_r': 'arm_add_ips',
            'arm_flex_r': 'arm_flex_ips',
            'arm_rot_r': 'arm_rot_ips',
            'elbow_flex_r': 'elbow_flex_ips',
            'pro_sup_r': 'pro_sup_ips',
            'arm_add_l': 'arm_add_contra',
            'arm_flex_l': 'arm_flex_contra',
            'arm_rot_l': 'arm_rot_contra',
            'elbow_flex_l': 'elbow_flex_contra',
            'pro_sup_l': 'pro_sup_contra',

            # Specific to the 6dof shoulder
            'sh_plane_elev_r': 'arm_add_ips',
            'sh_plane_elev_l': 'arm_add_contra',
            'sh_elev_r': 'arm_flex_ips',
            'sh_elev_l': 'arm_flex_contra',
            'sh_axial_rot_r': 'arm_rot_ips',
            'sh_axial_rot_l': 'arm_rot_contra',

        }
    elif leg == 'l':
        return {
            'hip_flexion_l': 'hip_flexion_ips',
            'hip_adduction_l': 'hip_adduction_ips',
            'hip_rotation_l': 'hip_rotation_ips',
            'knee_angle_l': 'knee_angle_ips',
            'ankle_angle_l': 'ankle_angle_ips',
            'subtalar_angle_l': 'subtalar_angle_ips',
            'mtp_angle_l': 'mtp_angle_ips',
            'hip_flexion_r': 'hip_flexion_contra',
            'hip_adduction_r': 'hip_adduction_contra',
            'hip_rotation_r': 'hip_rotation_contra',
            'knee_angle_r': 'knee_angle_contra',
            'ankle_angle_r': 'ankle_angle_contra',
            'subtalar_angle_r': 'subtalar_angle_contra',
            'mtp_angle_r': 'mtp_angle_contra',

            'arm_add_l': 'arm_add_ips',
            'arm_flex_l': 'arm_flex_ips',
            'arm_rot_l': 'arm_rot_ips',
            'elbow_flex_l': 'elbow_flex_ips',
            'pro_sup_l': 'pro_sup_ips',
            'arm_add_r': 'arm_add_contra',
            'arm_flex_r': 'arm_flex_contra',
            'arm_rot_r': 'arm_rot_contra',
            'elbow_flex_r': 'elbow_flex_contra',
            'pro_sup_r': 'pro_sup_contra',

            # Specific to the 6dof shoulder
            'sh_plane_elev_l': 'arm_add_ips',
            'sh_plane_elev_r': 'arm_add_contra',
            'sh_elev_l': 'arm_flex_ips',
            'sh_elev_r': 'arm_flex_contra',
            'sh_axial_rot_l': 'arm_rot_ips',
            'sh_axial_rot_r': 'arm_rot_contra',

        }
    else:
        raise ValueError(f"Invalid leg: {leg}")



def get_normalized_mahalanobis_distance(patient_mu_value, version="v01"):
    """
    Computes the normalized Mahalanobis distance between a sample and the healthy distribution.
    The normalization is done by squaring the Mahalanobis distance and dividing by latent dimension size,
    so the expected value is ~1 if the sample is from a healthy distribution.
    """
    healthy_cov_matrix = np.load(f"../gaitVAE/{version}/healthy_cov_matrix.npy")
    healthy_mu_mean = np.load(f"../gaitVAE/{version}/healthy_mean_mu.npy")

    inv_cov = np.linalg.pinv(healthy_cov_matrix)
    d = mahalanobis(patient_mu_value.flatten(), healthy_mu_mean.flatten(), inv_cov)

    # Normalize: square and divide by dimensionality
    normalized_distance = (d ** 2) / healthy_mu_mean.shape[0]
    return normalized_distance

def calculate_distance_to_healthy(trial_mus, version="v01"):
    return np.array([get_normalized_mahalanobis_distance(mu, version=version) for mu in trial_mus])

def load_vae(version="v01"):
    model_path = f"../gaitVAE/{version}/model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 24*32
    hidden_layers = [256, 128]
    latent_dim = 16
    mean = np.load(f"../gaitVAE/{version}/mean.npy")
    std = np.load(f"../gaitVAE/{version}/std.npy")
    normalize = (mean, std)

    vae = VariationalAutoEncoder(input_dim, hidden_layers, latent_dim, normalize=normalize).to(device)
    full_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    vae.load_state_dict(full_state_dict)
    vae.eval()
    return vae

def get_mu_from_df(df, model, device, zero_shoulder=False):
    df = np.array(df)  # Combine into one NumPy array

    if zero_shoulder:
        # Get indices of joints to zero
        joint_names_to_zero = ['arm_add_ips', 'arm_flex_ips', 'arm_add_contra', 'arm_flex_contra']
        indices_to_zero = [VAE_COLUMN_ORDER.index(name) for name in joint_names_to_zero]

        # Zero out the specified joints
        df[:, :, indices_to_zero] = 0

    df = torch.tensor(df, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        _, _, mu, _, _ = model(df)
    return mu.cpu().detach().numpy()


# The Encoder and Decoder classes are defined here
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, latent_dim, dropout=0.2):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_layers[i]))
            else:
                self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(nn.LayerNorm(hidden_layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        self.fc_mu = nn.Linear(hidden_layers[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_layers[-1], latent_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Sequentially apply each layer
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_layers, output_dim, dropout=0.2):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            if i == 0:
                self.layers.append(nn.Linear(latent_dim, hidden_layers[i]))
            else:
                self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(nn.LayerNorm(hidden_layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))  # Dropout after LayerNorm

        self.fc_out = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # Sequentially apply each layer
        x = self.fc_out(x)
        return x


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, z_dims, dropout=0.2, num_masked=0, timepoint_mask=0, c_gs=1, use_gs_regressor=True, normalize=(0, 1)):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_layers, z_dims, dropout)
        self.decoder = Decoder(z_dims, list(reversed(hidden_layers)), input_dim, dropout)
        self.num_masked = num_masked
        self.timepoint_mask = timepoint_mask
        self.c_gs = c_gs
        self.gait_speed_regressor = nn.Linear(z_dims, 1)
        self.use_gs_regressor = use_gs_regressor
        self.normalize_mean = normalize[0]
        self.normalize_std = normalize[1]

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, use_reparam=False):

        # Normalization
        x = (x - self.normalize_mean) / self.normalize_std

        x = x.view(x.shape[0], -1)
        mu, log_var = self.encoder(x)

        if use_reparam:
            z = self.reparameterization(mu, log_var)
        else:
            z = mu

        x_recon = self.decoder(z)
        x_recon = x_recon.view(x.shape[0], 24, -1)

        # Un-normalization
        x_recon = x_recon * self.normalize_std + self.normalize_mean

        # Gait speed distance
        norm_z_distance = torch.norm(mu, p=2, dim=1) / torch.sqrt(torch.tensor(mu.shape[1], device=mu.device, dtype=mu.dtype))

        if self.use_gs_regressor:
            pred_gait_speed = self.gait_speed_regressor(mu).squeeze(-1)
        else:
            pred_gait_speed = self.c_gs / (norm_z_distance + 1e-8)

        return x_recon, z, mu, log_var, pred_gait_speed