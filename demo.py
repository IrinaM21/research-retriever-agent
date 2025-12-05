from agent import Agent

agent = Agent()

query = "Can you summarize what's novel about this abstract?"

response = agent.run(query)

print(response)