import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "../integrations/supabase/client";

export default function Auth() {
  const navigate = useNavigate();

  useEffect(() => {
    const user = localStorage.getItem("nk_user");
    if (user) {
      navigate("/home");
    }
  }, [navigate]);

  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    if (!isLogin && password !== confirmPassword) {
      setError("Passwords do not match");
      setLoading(false);
      return;
    }

    try {
      let result;

      if (isLogin) {
        // Login using Supabase
        result = await supabase.auth.signInWithPassword({
          email,
          password,
        });
      } else {
        // Signup using Supabase
        result = await supabase.auth.signUp({
          email,
          password,
          options: {
            emailRedirectTo: `${window.location.origin}/auth`,
          },
        });
      }

      if (result.error) {
        throw result.error;
      }

      // If sign up succeeds but no session is returned, it means email confirmation is active
      if (!isLogin && !result.data.session) {
        setSuccess("Registration successful! Please check your email to verify your account.");
        setEmail("");
        setPassword("");
        setConfirmPassword("");
        setIsLogin(true); // Switch to login view
        return;
      }

      // Save user session
      localStorage.setItem("nk_user", JSON.stringify(result.data.user));
      navigate("/home");
    } catch (err: any) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm space-y-4 p-6 border rounded-lg shadow"
      >
        <h1 className="text-xl font-semibold text-center">
          {isLogin ? "Login" : "Sign Up"}
        </h1>

        {error && (
          <p className="text-sm text-red-500 text-center">{error}</p>
        )}

        {success && (
          <p className="text-sm text-green-600 text-center font-medium bg-green-50 p-2 rounded border border-green-200">{success}</p>
        )}

        <input
          type="email"
          placeholder="Email"
          className="w-full border p-2 rounded"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <input
          type="password"
          placeholder="Password"
          className="w-full border p-2 rounded"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        {!isLogin && (
          <input
            type="password"
            placeholder="Confirm Password"
            className="w-full border p-2 rounded"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
          />
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-primary text-white p-2 rounded disabled:opacity-50"
        >
          {loading ? "Please wait..." : isLogin ? "Login" : "Sign Up"}
        </button>

        <p className="text-center text-sm">
          {isLogin ? "New user?" : "Already have an account?"}{" "}
          <button
            type="button"
            onClick={() => {
              setIsLogin(!isLogin);
              setError("");
              setSuccess("");
              setPassword("");
              setConfirmPassword("");
            }}
            className="text-primary underline"
          >
            {isLogin ? "Sign up" : "Login"}
          </button>
        </p>
      </form>
    </div>
  );
}
