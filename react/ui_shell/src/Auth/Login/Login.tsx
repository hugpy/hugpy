import * as React from "react";
import { useEffect, useState } from "react";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import Stack from "@mui/material/Stack";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { Alert, Card, CardContent, Collapse } from "@mui/material";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "./../AuthProvider";

type RedirectState = {
  from?: {
    pathname?: string;
    search?: string;
    hash?: string;
  };
  justRegistered?: boolean;
  username?: string;
};


const Login: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { signIn } = useAuth();

  const redirectState = location.state as RedirectState | null;

  const redirectTo =
    `${redirectState?.from?.pathname || "/"}${redirectState?.from?.search || ""}${redirectState?.from?.hash || ""}`;

  // Post-registration banner. Cleared once the user submits.
  const [notice, setNotice] = useState<string | null>(
    redirectState?.justRegistered
      ? "Account created. You can sign in once an administrator approves it."
      : null
  );

  const [errors, setErrors] = useState(false);
  const [errorMessage, setErrorMessage] = useState("Credentials incorrect!");
  const [loading, setLoading] = useState(false);

  // Pre-fill username if we just came from /register.
  const [prefilledUsername] = useState(redirectState?.username || "");

  // Clear the post-register banner from history so a refresh doesn't re-show it.
  useEffect(() => {
    if (redirectState?.justRegistered) {
      window.history.replaceState({}, "");
    }
  }, [redirectState?.justRegistered]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setNotice(null);

    const data = new FormData(event.currentTarget);
    const username = String(data.get("username") || "").trim();
    const password = String(data.get("password") || "");

    if (!username || !password) {
      setErrorMessage("Username and password are required.");
      setErrors(true);
      return;
    }

    setLoading(true);
    setErrors(false);

    try {
      const ok = await signIn(username, password);

      if (!ok) {
        setErrorMessage("Credentials incorrect or sign-in not permitted.");
        setErrors(true);
        return;
      }

      navigate(redirectTo, { replace: true });
    } catch (error) {
      console.error("Login error:", error);
      setErrorMessage("Login request failed.");
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
          <Typography component="h1" variant="h5">Sign in</Typography>

          <Collapse in={!!notice}>
            <Alert severity="info" sx={{ mt: 2 }}>{notice}</Alert>
          </Collapse>

          <Collapse in={errors}>
            <Alert severity="error" sx={{ mt: 2 }}>{errorMessage}</Alert>
          </Collapse>

          <Box component="form" onSubmit={handleSubmit} noValidate sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="username"
              label="User Name"
              name="username"
              defaultValue={prefilledUsername}
              error={errors}
              autoComplete="username"
              autoFocus
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
              error={errors}
              autoComplete="current-password"
              disabled={loading}
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={loading}
              sx={{ mt: 3, mb: 2 }}
            >
              {loading ? "Signing in..." : "Sign In"}
            </Button>

            <Stack direction="row" spacing={2} sx={{ justifyContent: "flex-end", mt: 2 }}>
              <Link to="/register">Register</Link>
              <Link to="/change-password">Change Password</Link>
              <Link to="/">Back</Link>
            </Stack>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default Login;