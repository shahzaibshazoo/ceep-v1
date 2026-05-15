"""
CPML Implementation to add to BatchedFDTD2D

Add this method after _setup_cpml()
"""

def _apply_cpml(self):
    """
    Apply Convolutional PML corrections to fields.

    This implements the standard CPML formulation from Roden & Gedney (2000).
    Must be called after each field update in the main time loop.
    """
    cp = self.xp
    n = self.cpml_n
    nx, ny = self.nx, ny

    # ========================================================================
    # X-Direction PML (Left and Right boundaries)
    # ========================================================================

    # Left boundary (x = 0 to n-1)
    for i in range(n):
        # Compute field derivatives in PML region
        dEz_dy = (self.ez[:, i, 1:] - self.ez[:, i, :-1]) * self.inv_dy

        # Update psi (memory variable)
        self.psi_hyx_lo[:, i, :-1] = (
            self.cpml_b_y[i] * self.psi_hyx_lo[:, i, :-1]
            + self.cpml_c_y[i] * dEz_dy
        )

        # Apply correction to Hx
        self.hx[:, i, :-1] -= self.db[0, i, :-1] * self.psi_hyx_lo[:, i, :-1]

    # Right boundary (x = nx-n to nx-1)
    for i in range(n):
        x_idx = nx - n + i
        dEz_dy = (self.ez[:, x_idx, 1:] - self.ez[:, x_idx, :-1]) * self.inv_dy

        self.psi_hyx_hi[:, i, :-1] = (
            self.cpml_b_y[n-1-i] * self.psi_hyx_hi[:, i, :-1]
            + self.cpml_c_y[n-1-i] * dEz_dy
        )

        self.hx[:, x_idx, :-1] -= self.db[0, x_idx, :-1] * self.psi_hyx_hi[:, i, :-1]

    # ========================================================================
    # Y-Direction PML (Bottom and Top boundaries)
    # ========================================================================

    # Bottom boundary (y = 0 to n-1)
    for j in range(n):
        dEz_dx = (self.ez[:, 1:, j] - self.ez[:, :-1, j]) * self.inv_dx

        self.psi_hxy_lo[:, :-1, j] = (
            self.cpml_b_x[j] * self.psi_hxy_lo[:, :-1, j]
            + self.cpml_c_x[j] * dEz_dx
        )

        self.hy[:, :-1, j] += self.db[0, :-1, j] * self.psi_hxy_lo[:, :-1, j]

    # Top boundary (y = ny-n to ny-1)
    for j in range(n):
        y_idx = ny - n + j
        dEz_dx = (self.ez[:, 1:, y_idx] - self.ez[:, :-1, y_idx]) * self.inv_dx

        self.psi_hxy_hi[:, :-1, j] = (
            self.cpml_b_x[n-1-j] * self.psi_hxy_hi[:, :-1, j]
            + self.cpml_c_x[n-1-j] * dEz_dx
        )

        self.hy[:, :-1, y_idx] += self.db[0, :-1, y_idx] * self.psi_hxy_hi[:, :-1, j]

    # ========================================================================
    # Ez field PML corrections
    # ========================================================================

    # X-direction (left and right)
    for i in range(n):
        # Left
        curl_H_x = (self.hy[:, i+1, :] - self.hy[:, i, :]) * self.inv_dx
        self.psi_ezx_lo[:, i, :] = (
            self.cpml_b_x[i] * self.psi_ezx_lo[:, i, :]
            + self.cpml_c_x[i] * curl_H_x
        )
        self.ez[:, i, :] += self.cb[0, i, :] * self.psi_ezx_lo[:, i, :]

        # Right
        x_idx = nx - n + i
        curl_H_x = (self.hy[:, x_idx+1, :] - self.hy[:, x_idx, :]) * self.inv_dx if x_idx+1 < nx else 0
        if x_idx+1 < nx:
            self.psi_ezx_hi[:, i, :] = (
                self.cpml_b_x[n-1-i] * self.psi_ezx_hi[:, i, :]
                + self.cpml_c_x[n-1-i] * curl_H_x
            )
            self.ez[:, x_idx, :] += self.cb[0, x_idx, :] * self.psi_ezx_hi[:, i, :]

    # Y-direction (bottom and top)
    for j in range(n):
        # Bottom
        curl_H_y = -(self.hx[:, :, j+1] - self.hx[:, :, j]) * self.inv_dy
        self.psi_ezy_lo[:, :, j] = (
            self.cpml_b_y[j] * self.psi_ezy_lo[:, :, j]
            + self.cpml_c_y[j] * curl_H_y
        )
        self.ez[:, :, j] += self.cb[0, :, j] * self.psi_ezy_lo[:, :, j]

        # Top
        y_idx = ny - n + j
        curl_H_y = -(self.hx[:, :, y_idx+1] - self.hx[:, :, y_idx]) * self.inv_dy if y_idx+1 < ny else 0
        if y_idx+1 < ny:
            self.psi_ezy_hi[:, :, j] = (
                self.cpml_b_y[n-1-j] * self.psi_ezy_hi[:, :, j]
                + self.cpml_c_y[n-1-j] * curl_H_y
            )
            self.ez[:, :, y_idx] += self.cb[0, :, y_idx] * self.psi_ezy_hi[:, :, j]
