/**
 * Hook para manejar autenticación con Firebase
 */
import { useEffect, useState } from 'react';
import { onAuthChange, getCurrentUser, getIdToken } from '../firebase';
import type { User } from 'firebase/auth';

interface UseAuthReturn {
  user: User | null;
  loading: boolean;
  getToken: () => Promise<string | null>;
  isAuthenticated: boolean;
}

export function useAuth(): UseAuthReturn {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Verificar usuario actual inmediatamente
    const currentUser = getCurrentUser();
    setUser(currentUser);
    setLoading(false);

    // Suscribirse a cambios en el estado de autenticación
    const unsubscribe = onAuthChange((user) => {
      setUser(user);
      setLoading(false);
    });

    return unsubscribe; // Cleanup al desmontar
  }, []);

  const getToken = async (): Promise<string | null> => {
    return await getIdToken();
  };

  return {
    user,
    loading,
    getToken,
    isAuthenticated: !!user,
  };
}

