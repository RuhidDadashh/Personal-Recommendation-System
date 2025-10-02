// Recommendations System JavaScript

// Recommendation algorithms and data processing
class RecommendationEngine {
    constructor() {
        this.movieDatabase = [];
        this.userPreferences = {};
        this.algorithms = {
            collaborative: this.collaborativeFiltering.bind(this),
            contentBased: this.contentBasedFiltering.bind(this),
            hybrid: this.hybridFiltering.bind(this)
        };
    }

    // Initialize with movie data
    initialize(movieData) {
        this.movieDatabase = movieData;
        this.buildGenreMatrix();
    }

    // Build genre similarity matrix
    buildGenreMatrix() {
        this.genreMatrix = {};
        this.movieDatabase.forEach(movie => {
            const genres = movie.genre.split(', ');
            this.genreMatrix[movie.id] = genres;
        });
    }

    // Content-based filtering
    contentBasedFiltering(selectedMovieIds, count = 6) {
        const selectedGenres = this.extractGenres(selectedMovieIds);
        const genreWeights = this.calculateGenreWeights(selectedGenres);
        
        const recommendations = this.movieDatabase
            .filter(movie => !selectedMovieIds.includes(movie.id))
            .map(movie => ({
                ...movie,
                score: this.calculateContentScore(movie, genreWeights)
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, count);

        return this.formatRecommendations(recommendations);
    }

    // Collaborative filtering simulation
    collaborativeFiltering(selectedMovieIds, count = 6) {
        // Simulate user-item matrix and collaborative filtering
        const similarUsers = this.findSimilarUsers(selectedMovieIds);
        const recommendations = this.getCollaborativeRecommendations(similarUsers, selectedMovieIds);
        
        return recommendations.slice(0, count);
    }

    // Hybrid filtering (combines content-based and collaborative)
    hybridFiltering(selectedMovieIds, count = 6) {
        const contentRecs = this.contentBasedFiltering(selectedMovieIds, count * 2);
        const collabRecs = this.collaborativeFiltering(selectedMovieIds, count * 2);
        
        // Combine and weight recommendations
        const hybridRecs = this.combineRecommendations(contentRecs, collabRecs, 0.6, 0.4);
        
        return hybridRecs.slice(0, count);
    }

    // Extract genres from selected movies
    extractGenres(movieIds) {
        const genres = [];
        movieIds.forEach(id => {
            const movie = this.movieDatabase.find(m => m.id === id);
            if (movie) {
                genres.push(...movie.genre.split(', '));
            }
        });
        return genres;
    }

    // Calculate genre weights based on frequency
    calculateGenreWeights(genres) {
        const weights = {};
        genres.forEach(genre => {
            weights[genre] = (weights[genre] || 0) + 1;
        });
        
        // Normalize weights
        const total = Object.values(weights).reduce((sum, weight) => sum + weight, 0);
        Object.keys(weights).forEach(genre => {
            weights[genre] = weights[genre] / total;
        });
        
        return weights;
    }

    // Calculate content-based score for a movie
    calculateContentScore(movie, genreWeights) {
        const movieGenres = movie.genre.split(', ');
        let score = 0;
        
        movieGenres.forEach(genre => {
            if (genreWeights[genre]) {
                score += genreWeights[genre];
            }
        });
        
        // Add rating boost
        const rating = parseFloat(movie.rating.split('/')[0]);
        score += (rating / 10) * 0.3;
        
        return score;
    }

    // Simulate finding similar users (for collaborative filtering)
    findSimilarUsers(selectedMovieIds) {
        // Simulated user preferences (in real app, this would come from database)
        const userProfiles = [
            { id: 1, preferences: [1, 2, 3, 7, 9] },
            { id: 2, preferences: [4, 5, 6, 10, 11] },
            { id: 3, preferences: [2, 4, 8, 12] },
            { id: 4, preferences: [1, 3, 7, 9, 11] },
            { id: 5, preferences: [5, 6, 8, 10, 12] }
        ];
        
        // Calculate similarity with current user
        return userProfiles.map(user => ({
            ...user,
            similarity: this.calculateJaccardSimilarity(selectedMovieIds, user.preferences)
        })).sort((a, b) => b.similarity - a.similarity);
    }

    // Calculate Jaccard similarity
    calculateJaccardSimilarity(set1, set2) {
        const intersection = set1.filter(x => set2.includes(x)).length;
        const union = new Set([...set1, ...set2]).size;
        return intersection / union;
    }

    // Get collaborative recommendations
    getCollaborativeRecommendations(similarUsers, selectedMovieIds) {
        const recommendations = new Map();
        
        similarUsers.slice(0, 3).forEach(user => {
            user.preferences.forEach(movieId => {
                if (!selectedMovieIds.includes(movieId)) {
                    const current = recommendations.get(movieId) || 0;
                    recommendations.set(movieId, current + user.similarity);
                }
            });
        });
        
        return Array.from(recommendations.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([movieId]) => {
                const movie = this.movieDatabase.find(m => m.id === movieId);
                return movie ? this.formatSingleRecommendation(movie) : null;
            })
            .filter(rec => rec !== null);
    }

    // Combine content-based and collaborative recommendations
    combineRecommendations(contentRecs, collabRecs, contentWeight, collabWeight) {
        const combined = new Map();
        
        // Add content-based recommendations
        contentRecs.forEach((rec, index) => {
            const score = (contentRecs.length - index) * contentWeight;
            combined.set(rec.title, { ...rec, combinedScore: score });
        });
        
        // Add collaborative recommendations
        collabRecs.forEach((rec, index) => {
            const score = (collabRecs.length - index) * collabWeight;
            if (combined.has(rec.title)) {
                combined.get(rec.title).combinedScore += score;
            } else {
                combined.set(rec.title, { ...rec, combinedScore: score });
            }
        });
        
        return Array.from(combined.values())
            .sort((a, b) => b.combinedScore - a.combinedScore);
    }

    // Format recommendations for display
    formatRecommendations(recommendations) {
        return recommendations.map(rec => this.formatSingleRecommendation(rec));
    }

    // Format single recommendation
    formatSingleRecommendation(movie) {
        return {
            title: movie.title,
            genre: movie.genre,
            rating: movie.rating,
            reason: this.generateRecommendationReason(movie)
        };
    }

    // Generate explanation for why this movie was recommended
    generateRecommendationReason(movie) {
        const reasons = [
            `Similar to your taste in ${movie.genre.split(', ')[0]} movies`,
            `Highly rated ${movie.genre.split(', ')[0]} film`,
            `Popular among users with similar preferences`,
            `Great choice based on your movie selection`
        ];
        return reasons[Math.floor(Math.random() * reasons.length)];
    }
}

// Advanced recommendation features
class RecommendationAnalytics {
    constructor() {
        this.userInteractions = [];
        this.feedbackData = [];
    }

    // Track user interactions
    trackInteraction(userId, movieId, interactionType, timestamp = Date.now()) {
        this.userInteractions.push({
            userId,
            movieId,
            interactionType, // 'view', 'like', 'dislike', 'watch'
            timestamp
        });
    }

    // Get user recommendation history
    getUserRecommendationHistory(userId) {
        return this.userInteractions
            .filter(interaction => interaction.userId === userId)
            .sort((a, b) => b.timestamp - a.timestamp);
    }

    // Calculate recommendation accuracy
    calculateAccuracy(recommendations, userFeedback) {
        if (userFeedback.length === 0) return 0;
        
        const accurateRecommendations = userFeedback.filter(
            feedback => feedback.rating >= 4
        ).length;
        
        return (accurateRecommendations / userFeedback.length) * 100;
    }

    // Get trending movies
    getTrendingMovies(timeframe = 7) { // days
        const cutoffTime = Date.now() - (timeframe * 24 * 60 * 60 * 1000);
        const recentInteractions = this.userInteractions
            .filter(interaction => interaction.timestamp > cutoffTime);
        
        const movieCounts = {};
        recentInteractions.forEach(interaction => {
            movieCounts[interaction.movieId] = (movieCounts[interaction.movieId] || 0) + 1;
        });
        
        return Object.entries(movieCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);
    }

    // Generate diversity metrics
    calculateDiversityScore(recommendations) {
        const genres = recommendations.map(rec => rec.genre);
        const uniqueGenres = new Set(genres.flatMap(g => g.split(', ')));
        return uniqueGenres.size / genres.length;
    }
}

// Recommendation UI controller
class RecommendationUI {
    constructor(engine, analytics) {
        this.engine = engine;
        this.analytics = analytics;
        this.currentAlgorithm = 'hybrid';
    }

    // Set recommendation algorithm
    setAlgorithm(algorithm) {
        if (this.engine.algorithms[algorithm]) {
            this.currentAlgorithm = algorithm;
            return true;
        }
        return false;
    }

    // Get recommendations with current algorithm
    async getRecommendations(selectedMovieIds) {
        try {
            const recommendations = await this.engine.algorithms[this.currentAlgorithm](selectedMovieIds);
            
            // Track this recommendation request
            const userId = this.getCurrentUserId();
            if (userId) {
                selectedMovieIds.forEach(movieId => {
                    this.analytics.trackInteraction(userId, movieId, 'selected');
                });
            }
            
            return {
                success: true,
                recommendations,
                algorithm: this.currentAlgorithm,
                diversity: this.analytics.calculateDiversityScore(recommendations)
            };
        } catch (error) {
            console.error('Error generating recommendations:', error);
            return {
                success: false,
                error: 'Failed to generate recommendations'
            };
        }
    }

    // Get current user ID from session
    getCurrentUserId() {
        const userData = sessionStorage.getItem('currentUser');
        return userData ? JSON.parse(userData).id : null;
    }

    // Provide feedback on recommendation
    provideFeedback(movieTitle, rating, userId = null) {
        userId = userId || this.getCurrentUserId();
        if (userId) {
            this.analytics.feedbackData.push({
                userId,
                movieTitle,
                rating,
                timestamp: Date.now()
            });
        }
    }
}

// Initialize recommendation system
let recommendationEngine, recommendationAnalytics, recommendationUI;

function initRecommendationSystem(movieData) {
    recommendationEngine = new RecommendationEngine();
    recommendationAnalytics = new RecommendationAnalytics();
    recommendationUI = new RecommendationUI(recommendationEngine, recommendationAnalytics);
    
    recommendationEngine.initialize(movieData);
    
    return recommendationUI;
}

// Export for use in other files
window.RecommendationSystem = {
    RecommendationEngine,
    RecommendationAnalytics,
    RecommendationUI,
    initRecommendationSystem
};