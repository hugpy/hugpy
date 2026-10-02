import React, { useState } from "react";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Grid from "@mui/material/Grid";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { Alert, Card, CardContent, Collapse } from "@mui/material";
import Stack from "@mui/material/Stack";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthProvider";

function ChangePassword() {
  const navigate = useNavigate();
  const { changePassword } = useAuth();

  const [errors, setErrors] = useState(false);
  const [success, setSuccess] = useState(false);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const data = new FormData(event.currentTarget);

    const currentPassword = String(data.get("current_password") || "");
    const newPassword = String(data.get("new_password") || "");
    const verifyPassword = String(data.get("verify_password") || "");

    setErrors(false);
    setSuccess(false);
    setMessage("");

    if (!currentPassword || !newPassword || !verifyPassword) {
      setErrors(true);
      setMessage("All password fields are required.");
      return;
    }

    if (newPassword !== verifyPassword) {
      setErrors(true);
      setMessage("New password and verify password do not match.");
      return;
    }

    if (newPassword.length < 10) {
      setErrors(true);
      setMessage("New password must be at least 10 characters.");
      return;
    }

    setLoading(true);

    try {
      const res = await changePassword({ currentPassword, newPassword });

      if (!res.ok) {
        setErrors(true);
        setMessage(res.error || "There was an issue changing your password.");
        return;
      }

      // changePassword already re-validated the session via the context.
      setSuccess(true);
      setMessage("Password updated successfully.");

      setTimeout(() => {
        navigate("/", { replace: true });
      }, 800);
    } catch (error) {
      console.error("Change password error:", error);
      setErrors(true);
      setMessage("Change password request failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card style={{ marginTop: "1rem" }}>
      <CardContent>
        <Box
          sx={{
            marginTop: 8,
            marginBottom: 8,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
          }}
        >
          <Typography component="h1" variant="h5">
            Change Password
          </Typography>

          <Collapse in={errors}>
            <Alert severity="error">{message}</Alert>
          </Collapse>

          <Collapse in={success}>
            <Alert severity="success">{message}</Alert>
          </Collapse>

          <Box component="form" noValidate onSubmit={handleSubmit} sx={{ mt: 1 }}>
            <Grid container spacing={2}>
              <Grid size={{ xs: 12 }}>
                <TextField
                  required
                  fullWidth
                  name="current_password"
                  label="Current Password"
                  type="password"
                  id="CurrentPassword"
                  autoComplete="current-password"
                  error={errors}
                  disabled={loading}
                />
              </Grid>

              <Grid size={{ xs: 12 }}>
                <TextField
                  required
                  fullWidth
                  name="new_password"
                  label="New Password"
                  type="password"
                  id="NewPassword"
                  autoComplete="new-password"
                  error={errors}
                  disabled={loading}
                />
              </Grid>

              <Grid size={{ xs: 12 }}>
                <TextField
                  required
                  fullWidth
                  name="verify_password"
                  label="Verify Password"
                  type="password"
                  id="VerifyPassword"
                  autoComplete="new-password"
                  error={errors}
                  disabled={loading}
                />
              </Grid>
            </Grid>

            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={loading}
              sx={{ mt: 3, mb: 2 }}
            >
              {loading ? "Updating..." : "Change Password"}
            </Button>

            <Stack direction="row" spacing={2} sx={{ justifyContent: "flex-end", mt: 2 }}>
              <Link to="/login">Login</Link>
              <Link to="/">Back</Link>
            </Stack>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}

export default ChangePassword;