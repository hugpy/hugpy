import * as React from "react";
import { useState } from "react";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Stack from "@mui/material/Stack";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { Alert, Card, CardContent, Collapse } from "@mui/material";
import { Link, useNavigate } from "react-router-dom";

import { useAuth } from "../AuthProvider";

function Register() {
  const navigate = useNavigate();
  const { signUp } = useAuth();
  const [errors, setErrors] = useState(false);
  const [errorMessage, setErrorMessage] = useState(
    "There was an issue registering your account."
  );
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const data = new FormData(event.currentTarget);
    const username = String(data.get("username") || "").trim();
    const email = String(data.get("email") || "").trim();
    const password = String(data.get("password") || "");

    if (!username || !email || !password) {
      setErrorMessage("Username, email, and password are required.");
      setErrors(true);
      return;
    }

    setLoading(true);
    setErrors(false);

    try {
      const res = await signUp({ username, email, password });

      if (!res.ok) {
        setErrorMessage(res.error || "There was an issue registering your account.");
        setErrors(true);
        return;
      }

      // Hand off to /login with a banner; Login reads `justRegistered`.
      navigate("/login", {
        replace: true,
        state: { justRegistered: true, username },
      });
    } catch (e) {
      console.error("Register error:", e);
      setErrorMessage("Registration request failed.");
      setErrors(true);
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
          <Typography component="h1" variant="h5">Sign up</Typography>

          <Collapse in={errors}>
            <Alert severity="error">{errorMessage}</Alert>
          </Collapse>

          <Box component="form" noValidate onSubmit={handleSubmit} sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="username"
              label="Username"
              name="username"
              autoComplete="username"
              autoFocus
              error={errors}
              disabled={loading}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="Email Address"
              name="email"
              type="email"
              autoComplete="email"
              error={errors}
              disabled={loading}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="new-password"
              error={errors}
              disabled={loading}
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={loading}
              sx={{ mt: 3, mb: 2 }}
            >
              {loading ? "Creating account..." : "Sign Up"}
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

export default Register;