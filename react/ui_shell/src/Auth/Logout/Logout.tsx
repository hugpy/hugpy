import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../AuthProvider";

// Fire-and-navigate: clears the session via the auth context, then returns
// home. Pure UI — no redux, no direct API calls.
function Logout() {
  const navigate = useNavigate();
  const { signOut } = useAuth();

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await signOut();
      if (!cancelled) navigate("/", { replace: true });
    })();
    return () => {
      cancelled = true;
    };
  }, [signOut, navigate]);

  return null;
}

export default Logout;
