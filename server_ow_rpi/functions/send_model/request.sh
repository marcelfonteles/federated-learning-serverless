RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${RED}ATTENTION:${NC} change the mongo db URL."
echo -e "${RED}ATTENTION:${NC} this request was not implemented. Will return an error."

curl -X POST http://172.17.0.1:9001/api/23bc46b1-71f6-4ed5-8c54-816aa4f8c502/send_model \
   -H "Content-Type: application/json" \
   -d '{"client_id": 0}'